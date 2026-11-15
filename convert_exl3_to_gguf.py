#!/usr/bin/env python3
"""Convert an EXL3 (exllamav3) Qwen3.5-family checkpoint to a GGUF for this fork (see dev/exl3-scope.md).

EXL3 linear layers keep their packed trellis bytes unchanged as GGML_TYPE_EXL3_<K> weights, plus two F16 vectors per
weight, <name>.exl3_suh (input side, K values) and <name>.exl3_svh (output side, N values). Everything else (norms,
GDN parameters, the bf16 embedding) is converted like convert_hf_to_gguf.py does for Qwen3_5ForConditionalGeneration.

The GGUF metadata (hparams, tokenizer, chat template) is copied from a donor GGUF of the same base model, and every
produced tensor is checked against the donor's tensor list and shapes. Unquantized tensors are also compared with the
donor's values (they are the same base checkpoint), which checks the V-head reorder used on the EXL3 weights too.

    python convert_exl3_to_gguf.py <exl3 dir> <donor.gguf> <out.gguf> [--no-mtp] [--no-donor-values]
"""

from __future__ import annotations

import argparse
import json
import logging
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / 'gguf-py'))
import gguf  # noqa: E402

logger = logging.getLogger('exl3-to-gguf')

MUL1_MULTIPLIER = 0x83DCD12D

EXL3_TYPES = {k: getattr(gguf.GGMLQuantizationType, f'EXL3_{k}') for k in range(1, 9)}

# metadata keys of the donor that describe its own quantization, not the model
DROP_KEYS_PREFIX = ('quantize.', 'general.file_type', 'general.quantization_version', 'general.quantized_by',
                    'general.description', 'general.name', 'general.architecture', 'GGUF.')


class SafeTensors:
    """Memory-mapped view of a sharded safetensors checkpoint."""

    DTYPES = {'BF16': np.uint16, 'F16': np.float16, 'F32': np.float32, 'I16': np.int16, 'I32': np.int32,
              'U8': np.uint8, 'I8': np.int8}

    def __init__(self, model_dir: Path):
        self.entries: dict[str, tuple[Path, int, int, str, list[int]]] = {}
        self.maps: dict[Path, np.memmap] = {}
        files = sorted(model_dir.glob('*.safetensors'))
        files = [f for f in files if f.name.startswith('model')]
        if not files:
            raise SystemExit(f'no model*.safetensors in {model_dir}')
        for f in files:
            with open(f, 'rb') as fh:
                (n,) = struct.unpack('<Q', fh.read(8))
                header = json.loads(fh.read(n))
            base = 8 + n
            for name, info in header.items():
                if name == '__metadata__':
                    continue
                b0, b1 = info['data_offsets']
                self.entries[name] = (f, base + b0, b1 - b0, info['dtype'], info['shape'])

    def has(self, name: str) -> bool:
        return name in self.entries

    def raw(self, name: str) -> np.ndarray:
        """uint8 view of the tensor bytes"""
        f, off, nbytes, _, _ = self.entries[name]
        if f not in self.maps:
            self.maps[f] = np.memmap(f, dtype=np.uint8, mode='r')
        return self.maps[f][off:off + nbytes]

    def get(self, name: str) -> np.ndarray:
        _, _, _, dtype, shape = self.entries[name]
        return self.raw(name).view(self.DTYPES[dtype]).reshape(shape)

    def f32(self, name: str) -> np.ndarray:
        _, _, _, dtype, _ = self.entries[name]
        a = self.get(name)
        if dtype == 'BF16':
            return (a.astype(np.uint32) << 16).view(np.float32)
        return a.astype(np.float32)


def reorder_v_heads(n: int, num_k_heads: int, num_v_per_k: int, head_dim: int) -> np.ndarray:
    """perm[j] = source index of new position j: grouped V order [k][v_per_k][hd] -> tiled [v_per_k][k][hd]
    (convert_hf_to_gguf.py _reorder_v_heads)"""
    idx = np.arange(n).reshape(num_k_heads, num_v_per_k, head_dim)
    return idx.transpose(1, 0, 2).reshape(n)


def full_perm(total: int, start: int, sub: np.ndarray) -> np.ndarray:
    """identity on [0, start), sub + start after"""
    return np.concatenate([np.arange(start), sub + start])


def block_perm(perm: np.ndarray, block: int) -> np.ndarray:
    """perm must move whole aligned blocks; returns the source block of each destination block"""
    n = perm.shape[0]
    assert n % block == 0
    pb = perm.reshape(n // block, block)
    src = pb[:, 0]
    assert np.all(src % block == 0) and np.array_equal(pb, src[:, None] + np.arange(block)), 'not block-aligned'
    return src // block


class Exl3Weight:
    def __init__(self, st: SafeTensors, prefix: str):
        tr_name = prefix + '.trellis'
        _, _, _, dtype, shape = st.entries[tr_name]
        assert dtype == 'I16' and len(shape) == 3 and shape[2] % 16 == 0, (tr_name, dtype, shape)
        self.bits = shape[2] // 16
        assert 1 <= self.bits <= 8, (tr_name, shape)
        self.kt, self.nt = shape[0], shape[1]
        self.K, self.N = self.kt * 16, self.nt * 16
        # K % 256: the ggml EXL3 types count one 256-weight tile per block along K; N % 128: output Hadamard blocks
        assert self.K % 256 == 0 and self.N % 128 == 0, (tr_name, self.K, self.N)
        # 2D view of the tiles: [k tile, n tile, 32 * bits bytes]
        self.tiles = st.raw(tr_name).reshape(self.kt, self.nt, 32 * self.bits)

        if st.has(prefix + '.mul1'):
            mul1 = int(st.get(prefix + '.mul1').reshape(-1)[0].astype(np.int64)) & 0xffffffff
            assert mul1 == MUL1_MULTIPLIER, f'{prefix}: unsupported mul1 multiplier {mul1:#x}'
        else:
            raise SystemExit(f'{prefix}: only the mul1 codebook is supported (no .mul1 tensor)')
        if st.has(prefix + '.mcg'):
            raise SystemExit(f'{prefix}: mcg codebook is not supported')

        self.suh = self._scale(st, prefix, 'suh', 'su', self.K)
        self.svh = self._scale(st, prefix, 'svh', 'sv', self.N)
        for extra in ('.bias',):
            if st.has(prefix + extra):
                raise SystemExit(f'{prefix}{extra}: EXL3 bias is not supported')
        self.used = {prefix + s for s in ('.trellis', '.mul1', '.suh', '.svh', '.su', '.sv') if st.has(prefix + s)}

    @staticmethod
    def _scale(st: SafeTensors, prefix: str, name_h: str, name_packed: str, n: int) -> np.ndarray:
        if st.has(f'{prefix}.{name_h}'):
            v = st.get(f'{prefix}.{name_h}')
            assert v.dtype == np.float16 and v.size == n, (prefix, name_h, v.shape)
            return np.ascontiguousarray(v.reshape(n))
        if st.has(f'{prefix}.{name_packed}'):
            # packed sign bits, LSB first (exllamav3 LinearEXL3.unpack_bf)
            bits = st.get(f'{prefix}.{name_packed}').view(np.uint16).reshape(-1)
            expanded = ((bits[:, None] >> np.arange(16, dtype=np.uint16)) & 1).reshape(-1)
            assert expanded.size == n, (prefix, name_packed, expanded.size, n)
            return (1.0 - 2.0 * expanded.astype(np.float32)).astype(np.float16)
        raise SystemExit(f'{prefix}: no {name_h} / {name_packed}')

    def permute_out(self, perm: np.ndarray) -> None:
        """y_new[j] = y_old[perm[j]] (whole 128-column blocks)"""
        self.tiles = self.tiles[:, np.repeat(block_perm(perm, 128) * 8, 8) + np.tile(np.arange(8), self.N // 128), :]
        self.svh = np.ascontiguousarray(self.svh[perm])

    def permute_in(self, perm: np.ndarray) -> None:
        """x_new[j] = x_old[perm[j]] (whole 128-row blocks)"""
        self.tiles = self.tiles[np.repeat(block_perm(perm, 128) * 8, 8) + np.tile(np.arange(8), self.K // 128), :, :]
        self.suh = np.ascontiguousarray(self.suh[perm])


def copy_metadata(writer: gguf.GGUFWriter, donor: gguf.GGUFReader, n_layer_total: int, drop_mtp: bool) -> None:
    for field in donor.fields.values():
        name = field.name
        if name.startswith(DROP_KEYS_PREFIX):
            continue
        if drop_mtp and name == 'qwen35.nextn_predict_layers':
            continue
        vtype = field.types[0]
        value = field.contents()
        if name == 'qwen35.block_count':
            value = n_layer_total
        if vtype == gguf.GGUFValueType.ARRAY:
            writer.add_key_value(name, value, vtype, sub_type=field.types[-1])
        else:
            writer.add_key_value(name, value, vtype)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('exl3_dir', type=Path)
    ap.add_argument('donor', type=Path, help='GGUF of the same base model (metadata + tensor list check)')
    ap.add_argument('out', type=Path)
    ap.add_argument('--no-mtp', action='store_true', help='leave out the MTP head (blk.<n_layer>)')
    ap.add_argument('--no-donor-values', action='store_true', help='skip the value check of unquantized tensors')
    ap.add_argument('--name', default=None, help='general.name (default: <exl3 dir name>)')
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    cfg = json.loads((args.exl3_dir / 'config.json').read_text(encoding='utf-8'))
    tc = cfg.get('text_config', cfg)
    qc = cfg.get('quantization_config', {})
    if qc.get('quant_method') != 'exl3':
        raise SystemExit('config.json: quantization_config.quant_method is not exl3')
    if qc.get('codebook', 'mul1') != 'mul1':
        raise SystemExit(f"codebook {qc.get('codebook')} is not supported (mul1 only)")

    n_layer = tc['num_hidden_layers']
    n_mtp = 0 if args.no_mtp else tc.get('mtp_num_hidden_layers', 0)
    n_k = tc['linear_num_key_heads']
    n_v = tc['linear_num_value_heads']
    hk = tc['linear_key_head_dim']
    hv = tc['linear_value_head_dim']
    v_per_k = n_v // n_k
    layer_types = tc['layer_types']

    donor = gguf.GGUFReader(args.donor)
    dk = {f.name: f for f in donor.fields.values()}

    def dkv(key):
        return dk[key].contents() if key in dk else None

    # the donor must describe the same model
    checks = {
        'qwen35.embedding_length': tc['hidden_size'],
        'qwen35.feed_forward_length': tc['intermediate_size'],
        'qwen35.attention.head_count': tc['num_attention_heads'],
        'qwen35.attention.head_count_kv': tc['num_key_value_heads'],
        'qwen35.ssm.state_size': hk,
        'qwen35.ssm.group_count': n_k,
        'qwen35.ssm.time_step_rank': n_v,
        'qwen35.ssm.inner_size': hv * n_v,
        'qwen35.full_attention_interval': tc.get('full_attention_interval', 4),
    }
    for key, want in checks.items():
        got = dkv(key)
        if got != want:
            raise SystemExit(f'donor mismatch: {key} = {got}, config.json says {want}')
    donor_layers = dkv('qwen35.block_count')
    donor_mtp = dkv('qwen35.nextn_predict_layers') or 0
    if donor_layers != n_layer + donor_mtp:
        raise SystemExit(f'donor block_count {donor_layers} != {n_layer} + {donor_mtp}')
    if n_mtp and donor_mtp != n_mtp:
        raise SystemExit(f'donor has {donor_mtp} MTP layers, checkpoint {n_mtp} (use --no-mtp or a -mtp donor)')
    donor_tensors = {t.name: t for t in donor.tensors}

    st = SafeTensors(args.exl3_dir)
    used: set[str] = set()

    out_tensors: list[tuple[str, np.ndarray, gguf.GGMLQuantizationType | None]] = []
    exl3_names: list[str] = []
    value_checks: list[tuple[str, np.ndarray]] = []

    def add_plain(gname: str, arr: np.ndarray, check: bool = True) -> None:
        out_tensors.append((gname, arr, None))
        if check:
            value_checks.append((gname, arr))

    def add_exl3(gname_base: str, w: Exl3Weight) -> None:
        tiles = np.ascontiguousarray(w.tiles)
        # byte rows of K * bits / 8 so gguf-py derives the shape [K, N]; the bytes are tile-ordered, not row-ordered
        out_tensors.append((gname_base + '.weight', tiles.reshape(w.N, w.K * w.bits // 8), EXL3_TYPES[w.bits]))
        out_tensors.append((gname_base + '.exl3_suh', w.suh, None))
        out_tensors.append((gname_base + '.exl3_svh', w.svh, None))
        exl3_names.append(gname_base + '.weight')
        used.update(w.used)

    def norm(name: str, plus_one: bool) -> np.ndarray:
        used.add(name)
        v = st.f32(name)
        return v + 1.0 if plus_one else v

    def plain_f16(name: str) -> np.ndarray:
        used.add(name)
        a = st.get(name)
        if a.dtype == np.float16:
            return np.ascontiguousarray(a)
        return st.f32(name)

    v_perm_heads = reorder_v_heads(n_v * hv, n_k, v_per_k, hv)
    v_perm_scalar = reorder_v_heads(n_v, n_k, v_per_k, 1)

    def full_attn_block(src: str, bid: int) -> None:
        for hf, gg in (('self_attn.q_proj', 'attn_q'), ('self_attn.k_proj', 'attn_k'), ('self_attn.v_proj', 'attn_v'),
                       ('self_attn.o_proj', 'attn_output')):
            add_exl3(f'blk.{bid}.{gg}', Exl3Weight(st, f'{src}.{hf}'))
        add_plain(f'blk.{bid}.attn_q_norm.weight', norm(f'{src}.self_attn.q_norm.weight', True))
        add_plain(f'blk.{bid}.attn_k_norm.weight', norm(f'{src}.self_attn.k_norm.weight', True))

    def common_block(src: str, bid: int) -> None:
        add_plain(f'blk.{bid}.attn_norm.weight', norm(f'{src}.input_layernorm.weight', True))
        add_plain(f'blk.{bid}.post_attention_norm.weight', norm(f'{src}.post_attention_layernorm.weight', True))
        for hf, gg in (('mlp.gate_proj', 'ffn_gate'), ('mlp.up_proj', 'ffn_up'), ('mlp.down_proj', 'ffn_down')):
            add_exl3(f'blk.{bid}.{gg}', Exl3Weight(st, f'{src}.{hf}'))

    lm = 'model.language_model'
    for il in range(n_layer):
        src = f'{lm}.layers.{il}'
        common_block(src, il)
        if layer_types[il] == 'full_attention':
            full_attn_block(src, il)
            continue
        assert layer_types[il] == 'linear_attention', layer_types[il]
        la = f'{src}.linear_attn'

        qkv = Exl3Weight(st, f'{la}.in_proj_qkv')
        assert qkv.N == 2 * n_k * hk + n_v * hv, qkv.N
        qkv.permute_out(full_perm(qkv.N, 2 * n_k * hk, v_perm_heads))
        add_exl3(f'blk.{il}.attn_qkv', qkv)

        z = Exl3Weight(st, f'{la}.in_proj_z')
        z.permute_out(v_perm_heads)
        add_exl3(f'blk.{il}.attn_gate', z)

        out = Exl3Weight(st, f'{la}.out_proj')
        out.permute_in(v_perm_heads)
        add_exl3(f'blk.{il}.ssm_out', out)

        for hf, gg in (('in_proj_a', 'ssm_alpha'), ('in_proj_b', 'ssm_beta')):
            w = plain_f16(f'{la}.{hf}.weight')
            assert w.shape == (n_v, tc['hidden_size']), (hf, w.shape)
            add_plain(f'blk.{il}.{gg}.weight', np.ascontiguousarray(w[v_perm_scalar]))

        used.add(f'{la}.A_log')
        add_plain(f'blk.{il}.ssm_a', np.ascontiguousarray(-np.exp(st.f32(f'{la}.A_log'))[v_perm_scalar]))
        used.add(f'{la}.dt_bias')
        add_plain(f'blk.{il}.ssm_dt.bias', np.ascontiguousarray(st.f32(f'{la}.dt_bias')[v_perm_scalar]))

        used.add(f'{la}.conv1d.weight')
        conv = st.f32(f'{la}.conv1d.weight').reshape(-1, tc['linear_conv_kernel_dim'])
        qk_ch = 2 * n_k * hk
        conv = np.concatenate([conv[:qk_ch], conv[qk_ch:][v_perm_heads]], axis=0)
        add_plain(f'blk.{il}.ssm_conv1d.weight', np.ascontiguousarray(conv))

        add_plain(f'blk.{il}.ssm_norm.weight', norm(f'{la}.norm.weight', False))

    # MTP head as blk.<n_layer> (full-attention block + nextn tensors)
    for im in range(n_mtp):
        bid = n_layer + im
        src = f'mtp.layers.{im}'
        common_block(src, bid)
        full_attn_block(src, bid)
        add_exl3(f'blk.{bid}.nextn.eh_proj', Exl3Weight(st, 'mtp.fc'))
        add_plain(f'blk.{bid}.nextn.enorm.weight', norm('mtp.pre_fc_norm_embedding.weight', True))
        add_plain(f'blk.{bid}.nextn.hnorm.weight', norm('mtp.pre_fc_norm_hidden.weight', True))
        add_plain(f'blk.{bid}.nextn.shared_head_norm.weight', norm('mtp.norm.weight', True))

    add_plain('output_norm.weight', norm(f'{lm}.norm.weight', True))
    add_exl3('output', Exl3Weight(st, 'lm_head'))

    emb = f'{lm}.embed_tokens.weight'
    _, _, _, emb_dtype, emb_shape = st.entries[emb]
    assert emb_dtype == 'BF16' and emb_shape == [tc['vocab_size'], tc['hidden_size']], (emb_dtype, emb_shape)
    used.add(emb)
    out_tensors.append(('token_embd.weight', st.raw(emb).reshape(emb_shape[0], emb_shape[1] * 2),
                        gguf.GGMLQuantizationType.BF16))

    # every checkpoint tensor except the vision tower must have been consumed
    leftover = sorted(n for n in st.entries if n not in used and '.visual.' not in n and not n.startswith('model.visual'))
    if n_mtp == 0:
        leftover = [n for n in leftover if not n.startswith('mtp.')]
    if leftover:
        raise SystemExit(f'unconverted checkpoint tensors ({len(leftover)}), e.g. {leftover[:8]}')

    # tensor list and shapes must match the donor (the EXL3 vectors are extra)
    produced = {name: arr for name, arr, _ in out_tensors}
    expected = {n for n in donor_tensors if n_mtp or not n.startswith(f'blk.{n_layer}.')}
    names = {n for n in produced if '.exl3_s' not in n}
    if names != expected:
        raise SystemExit(f'tensor list differs from the donor: missing {sorted(expected - names)[:8]}, '
                         f'extra {sorted(names - expected)[:8]}')
    raw_types = {name: rt for name, _, rt in out_tensors}
    for name in names:
        dshape = [int(x) for x in donor_tensors[name].shape]   # ggml order: [ne0, ne1, ...]
        arr = produced[name]
        rt = raw_types[name]
        if rt is not None:
            blck, tsize = gguf.GGML_QUANT_SIZES[rt]
            shape = list(arr.shape[:-1]) + [arr.shape[-1] // tsize * blck]
        else:
            shape = list(arr.shape)
        shape = shape[::-1]
        while len(dshape) > len(shape) and dshape[-1] == 1:
            dshape.pop()
        if shape != dshape:
            raise SystemExit(f'{name}: shape {shape} != donor {dshape}')

    if not args.no_donor_values:
        worst = 0.0
        for name, arr in value_checks:
            dt = donor_tensors[name]
            if dt.tensor_type not in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
                continue
            ref = np.asarray(dt.data, dtype=np.float32).reshape(arr.shape)
            err = float(np.max(np.abs(ref - arr.astype(np.float32))) / (np.max(np.abs(ref)) + 1e-12))
            worst = max(worst, err)
            if err > 1e-2:
                raise SystemExit(f'{name}: differs from the donor (relative max error {err:.3g}) - wrong transform?')
        logger.info(f'unquantized tensors match the donor (worst relative max error {worst:.3g})')

    writer = gguf.GGUFWriter(args.out, 'qwen35')
    copy_metadata(writer, donor, n_layer + n_mtp, drop_mtp=n_mtp == 0)
    writer.add_name(args.name or args.exl3_dir.name)
    writer.add_string('exl3.version', str(qc.get('version', '')))
    writer.add_float32('exl3.bits', float(qc.get('bits', 0.0)))
    writer.add_string('exl3.codebook', 'mul1')

    nbytes = {'exl3': 0, 'other': 0}
    for name, arr, rt in out_tensors:
        writer.add_tensor(name, arr, raw_dtype=rt)
        nbytes['exl3' if rt in EXL3_TYPES.values() else 'other'] += arr.nbytes
    logger.info(f'{len(out_tensors)} tensors ({len(exl3_names)} EXL3 weights, {nbytes["exl3"] / 2**20:.0f} MiB; '
                f'other {nbytes["other"] / 2**20:.0f} MiB) -> {args.out}')

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


if __name__ == '__main__':
    main()
