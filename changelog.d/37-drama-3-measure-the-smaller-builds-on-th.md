### Added

- [`results/smaller-builds`](results/smaller-builds/README.md): the 5,900-expert quant on the Mac with its head
  trimmed, its codec in BF16 and its routed experts in int3 (GPTQ on the GB10). The int3 build wins at 14.50 GiB
  against 18.34 GiB: reference-voice WER 1.78 % against 1.50 %, built-in 5.78 % against 6.91 %, failed takes 0 and
  4 against 0 and 6 (48 takes per voice). Its listening page and audio stay on the Mac, unpublished.
