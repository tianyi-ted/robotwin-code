# EDULITE-A3 SDK local patch

The collector uses local additions to the upstream EDULITE-A3 SDK.

- Upstream repository: `https://github.com/RobStride/EDULITE_A3.git`
- Tested base commit: `ea7231f784ebb37e4c4120f7be8e3670514dc9ee`
- Patch: `EDULITE_A3_ea7231f_teaching.patch`

Apply from a clean EDULITE-A3 checkout at the base commit:

```bash
git checkout ea7231f784ebb37e4c4120f7be8e3670514dc9ee
git apply /path/to/robotwin_code/robot/edulite_a3_collect/sdk_patches/EDULITE_A3_ea7231f_teaching.patch
```

The patch adds the verified L7 teaching/passive feedback APIs used by
`hardware.py`; it does not contain collected data or motor calibration files.
