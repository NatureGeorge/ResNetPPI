# Usage

## Scripts

Make sure the scripts have execution permission: `chmod +x demo.sh`.

### `rcsb_batch_download.sh`

> <https://www.rcsb.org/docs/programmatic-access/batch-downloads-with-shell-script>

```bash
./rcsb_batch_download.sh -f pdb_list.txt -c
```

### `mask_msa.sh`

> <https://github.com/RosettaCommons/trRosetta2/blob/main/scripts/make_msa.sh>

```bash
make_msa.sh $in_fasta $out_dir $CPU $MEM $DB > $out_dir/log/make_msa.stdout 2> $out_dir/log/make_msa.stderr
```

### `msa.py`

```bash
nohup python msa.py -pdb_dir ./pdb/data/structures/divided/mmCIF/ \
                    -sc_dir ./src \
                    -n_cpu 10 \
                    -n_mem 16 \
                    -n_job 3 \
                    exp_pdb_464.csv ./wdir > run_msa.out &
```

### `training_pipeline.py`

```bash
python training_pipeline.py -h
```

### `inference_pipeline.py`

```bash
python inference_pipeline.py -h
```

## Training Pipeline

1. prepare PDB metadata (see ../example/pdb_list_train.csv and ../example/pdb_list_val.csv)
2. download PDB coordinate files via `rcsb_batch_download.sh`
3. download sequence database (e.g. [UniRef30_2020_06_hhsuite](http://wwwuser.gwdg.de/~compbiol/uniclust/2020_06/UniRef30_2020_06_hhsuite.tar.gz)) as well as corresponding tool (e.g. install `hhsuite` via conda)
4. run `msa.py`
5. run `training_pipeline.py`
