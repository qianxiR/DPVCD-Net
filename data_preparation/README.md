# LBFD-CD split generation

`build_lbfd_cd.py` scans the prepared LBFD-CD dataset in the repository root
and generates six category-specific split files under `splits/`.

The script reads matching PNG filenames from:

```text
train/{t1,t2,label}/
val/{t1,t2,label}/
test/{t1,t2,label}/
```

Samples prefixed with `LBFD_jiuzaigou_`, `LBFD_shimen_`, or
`LBFD_taitung_` are assigned to the landslide category. Samples prefixed with
`LBFD_whu_` are assigned to the building category. Each output line contains a
sample basename without the `.png` extension.

The current prepared dataset records the paper's category-wise `7:2:1` random
split with seed `42`; this script exports those existing assignments rather
than reshuffling the samples.

## Generate split files

Run from the LBFD-CD repository root:

```bash
python data_preparation/build_lbfd_cd.py
```

To verify that the checked-in files still match the current dataset without
rewriting them:

```bash
python data_preparation/build_lbfd_cd.py --check
```

The script verifies that every sample has matching `t1`, `t2`, and `label`
files, that no sample basename is shared across train/validation/test, and that
the resulting counts match the paper:

| Category | Train | Validation | Test | Total |
| --- | ---: | ---: | ---: | ---: |
| Landslide | 575 | 161 | 83 | 819 |
| Building | 1,815 | 522 | 259 | 2,596 |
| **Total** | **2,390** | **683** | **342** | **3,415** |

The split is patch-level and is not geographically disjoint. No identical
patch is shared among the training, validation, and test sets.
