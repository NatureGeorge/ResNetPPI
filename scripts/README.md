# Usage

Make sure the scripts have execution permission: `chmod +x demo.sh`.

## `rcsb_batch_download.sh`

> <https://www.rcsb.org/docs/programmatic-access/batch-downloads-with-shell-script>

```bash
./rcsb_batch_download.sh -f pdb_list.txt -c
```

## `mask_msa.sh`

> <https://github.com/RosettaCommons/trRosetta2/blob/main/scripts/make_msa.sh>

```bash
make_msa.sh $in_fasta $out_dir $CPU $MEM $DB > $out_dir/log/make_msa.stdout 2> $out_dir/log/make_msa.stderr
```
