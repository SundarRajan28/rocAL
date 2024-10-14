# torchrun --standalone --nproc_per_node 8 --rdzv_backend c10d train.py --dataset celeba --distributed --root /media/celeba --chkpt-dir chkpts/celeba_baseline_run/ 2>&1 | tee celeba_baseline_train_log.txt
# torchrun --standalone --nproc_per_node 8 --rdzv_backend c10d train.py --dataset celeba --distributed --root /media/celeba --rocal-cpu --chkpt-dir chkpts/celeba_rocal_cpu_run2/ 2>&1 | tee celeba_rocal_cpu_train_log1.txt
# torchrun --standalone --nproc_per_node 8 --rdzv_backend c10d train.py --dataset celeba --distributed --root /media/celeba --rocal-gpu --chkpt-dir chkpts/celeba_rocal_gpu_run3/ 2>&1 | tee celeba_rocal_gpu_train_log3.txt
# rm -rf ./images/eval/celeba/celeba_600_baseline
# python generate.py --dataset celeba --chkpt-path ./chkpts/celeba_baseline_run/celeba/celeba_600.pt --use-ddim --skip-schedule quadratic --subseq-size 100 --suffix _baseline --num-gpus 8
# rm -rf ./images/eval/celeba/celeba_600_rocal_cpu
# python generate.py --dataset celeba --chkpt-path ./chkpts/celeba_rocal_cpu_run2/celeba/celeba_600.pt --use-ddim --skip-schedule quadratic --subseq-size 100 --suffix _rocal_cpu --num-gpus 8
# rm -rf ./images/eval/celeba/celeba_600_rocal_gpu
# python generate.py --dataset celeba --chkpt-path ./chkpts/celeba_rocal_gpu_run3/celeba/celeba_600.pt --use-ddim --skip-schedule quadratic --subseq-size 100 --suffix _rocal_gpu --num-gpus 8
# python eval.py --dataset celeba --sample-folder ./images/eval/celeba/celeba_600_baseline --root /media/celeba --metrics fid
# python eval.py --dataset celeba --sample-folder ./images/eval/celeba/celeba_600_rocal_cpu --root /media/celeba --metrics fid
# python eval.py --dataset celeba --sample-folder ./images/eval/celeba/celeba_600_rocal_gpu --root /media/celeba --metrics fid

CURRENTDATE=`date +"%Y-%m-%d-%T"`
loader=${LOADER:-"baseline"}
export TQDM_MININTERVAL=5

echo "Clear page cache"
sync && /sbin/sysctl vm.drop_caches=3
dir_name=logs/${loader}/${CURRENTDATE} 
mkdir -p $dir_name
torchrun --standalone --nproc_per_node 8 --rdzv_backend c10d train.py --dataset celeba --distributed --root /media/celeba --chkpt-dir $dir_name --image-dir $dir_name --loader $loader 2>&1 | tee -a $dir_name/out.log
python generate.py --dataset celeba --chkpt-path $dir_name/celeba/celeba_600.pt --use-ddim --skip-schedule quadratic --subseq-size 100 --num-gpus 8 --save-dir $dir_name 2>&1 | tee -a $dir_name/out.log
python eval.py --dataset celeba --sample-folder $dir_name/eval/celeba/celeba_600 --root /media/celeba --metrics fid 2>&1 | tee -a $dir_name/out.log
