import os
from dataclasses import dataclass, field
from tempfile import mkstemp
from spatpy.formats import mix_matrix
from subprocess import check_call
from spatpy.io import write_wav_file, read_wav_file, packaged_binary

# Commit at which repository https://forge.3gpp.org/rep/ivas-codec-pc/ivas-codec was built
DEFAULT_IVAS_GIT_HASH = 'ccf6ae2'
DEFAULT_IVAS_COD_BINARY = packaged_binary("ivas-codec", f"IVAS_cod_{DEFAULT_IVAS_GIT_HASH}")
DEFAULT_IVAS_DEC_BINARY = packaged_binary("ivas-codec", f"IVAS_dec_{DEFAULT_IVAS_GIT_HASH}")


@dataclass
class IVASCodec:
    bitrate: int = 256000
    intermediate_format: str = field(default_factory=lambda: "HOA1S")
    cod_binary: str = field(default_factory=lambda: DEFAULT_IVAS_COD_BINARY)
    dec_binary: str = field(default_factory=lambda: DEFAULT_IVAS_DEC_BINARY)

    def __post_init__(self):
        # only know how to do this for now
        assert self.intermediate_format == "HOA1S"

    def encode_options(self, fs):
        opt = []
        if self.intermediate_format == "HOA1S":
            opt += ["-sba", "+1"]
        opt.append(str(self.bitrate))
        opt.append(str(fs // 1000))
        return opt

    def decode_options(self, fs):
        opt = []
        if self.intermediate_format == "HOA1S":
            opt += ["FOA"]
        opt.append(str(fs // 1000))
        return opt

    def encode(self, pcm, pcm_format, fs, bitstream_file=None):
        _, tmp_wav = mkstemp(suffix=".wav")
        tmp_bitstream = None
        if bitstream_file is None:
            _, tmp_bitstream = mkstemp(suffix=".pkt")
            bitstream_file = tmp_bitstream
        bitstream = None
        try:
            pcm = mix_matrix(pcm_format, self.intermediate_format) @ pcm.T
            write_wav_file(tmp_wav, pcm, fs, channel_axis=0)
            check_call(
                [
                    self.cod_binary,
                ]
                + self.encode_options(fs)
                + [
                    tmp_wav,
                    bitstream_file,
                ]
            )
            with open(bitstream_file, "rb") as fobj:
                bitstream = fobj.read()
        finally:
            if tmp_bitstream is not None:
                os.remove(tmp_bitstream)
            os.remove(tmp_wav)
        return bitstream

    def decode(self, bitstream, pcm_format, fs):
        tmp_bitstream = None
        _, tmp_wav = mkstemp(suffix=".wav")
        try:
            if not isinstance(bitstream, str):
                assert isinstance(bitstream, bytearray)
                _, tmp_bitstream = mkstemp(suffix=".pkt")
                with open(tmp_bitstream, "wb") as fobj:
                    fobj.write(bitstream)
                bitstream = tmp_bitstream

            check_call(
                [
                    self.dec_binary,
                ]
                + self.decode_options(fs)
                + [
                    bitstream,
                    tmp_wav,
                ]
            )
            dec_pcm, fs = read_wav_file(tmp_wav, channel_axis=0)
        finally:
            if tmp_bitstream is not None:
                os.remove(tmp_bitstream)
            os.remove(tmp_wav)
        dec_pcm = mix_matrix(self.intermediate_format, pcm_format) @ dec_pcm
        return dec_pcm.T, fs

    def roundtrip(self, pcm, pcm_format, fs):
        _, tmp_bitstream = mkstemp(suffix=".pkt")
        try:
            self.encode(pcm, pcm_format, fs, bitstream_file=tmp_bitstream)
            dec_pcm, fs = self.decode(tmp_bitstream, pcm_format, fs)
        finally:
            os.remove(tmp_bitstream)
        return dec_pcm
