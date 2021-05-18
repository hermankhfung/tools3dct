#!/bin/bash
#SBATCH -p htc
#SBATCH -t 2:0:0
#SBATCH --mem=60GB
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --array=1-14%8

module load X11
module load Python

# Specify working directory
GSWORKDIR='/g/scb/mahamid/fung/Deformation'

# Provide rootname of before/after images and mask in each line of sitelist.inp
# Files of the following syntax expected: ROOTNAME_before.tif, ROOTNAME_after.tif, ROOTNAME_after_mask.tif
# ROOTNAME_after_mask.tif: mask image with regions to be excluded from the registration set to 0
GSSITE=$(cat $GSWORKDIR/sitelist.inp | head -${SLURM_ARRAY_TASK_ID} | tail -1)

GSROOT=$GSWORKDIR/$GSSITE

# Elastic registration, check Fiji path
/struct/cmueller/fung/bin/Fiji.app/java/linux-amd64/jdk1.8.0_172/jre/bin/java -Dplugins.dir=/struct/cmueller/fung/bin/Fiji.app/plugins -cp /struct/cmueller/fung/bin/Fiji.app/jars/ij-1.53c.jar:/struct/cmueller/fung/bin/Fiji.app/plugins/bUnwarpJ_-2.6.13.jar bunwarpj.bUnwarpJ_ -align ${GSROOT}_after.tif ${GSROOT}_after_mask.tif ${GSROOT}_before.tif NULL 2 5 0 0.1 0.1 1 30 ${GSROOT}_before_registered.tif ${GSROOT}_after_registered.tif -save_transformation

# Convert elastic transformation output to raw format, check Fiji path
/struct/cmueller/fung/bin/Fiji.app/java/linux-amd64/jdk1.8.0_172/jre/bin/java -Dplugins.dir=/struct/cmueller/fung/bin/Fiji.app/plugins -cp /struct/cmueller/fung/bin/Fiji.app/jars/ij-1.53c.jar:/struct/cmueller/fung/bin/Fiji.app/plugins/bUnwarpJ_-2.6.13.jar bunwarpj.bUnwarpJ_ -convert_to_raw ${GSROOT}_before.tif ${GSROOT}_after.tif ${GSROOT}_after_registered_transf.txt ${GSROOT}_rawtransf.txt

# Extract x- and y-displacements from transformation output in raw format
python $GSWORKDIR/process_raw_transf.py --input $GSROOT

rm ${GSROOT}_rawtransf.txt
