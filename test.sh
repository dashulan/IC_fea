for SEED in 42 1242 4830 3528 5660
do
  python src/main.py --nepochs 160 --lr 0.001 --wd 0  --batch_size 64 \
          --datasets cifar100 --network resnet18 --approach lwf_fea_CosLR \
          --exp-name lwf_fea_${SEED} --num-tasks 10 --seed $SEED
done

for SEED in 42 1242 4830 3528 5660
do
  python src/main.py --nepochs 160 --lr 0.001 --wd 0  --batch_size 64 \
          --datasets cifar100 --network resnet18 --approach lwf_fea_CosLR \
          --exp-name lwf_fea_${SEED} --num-tasks 20 --seed $SEED
done

#python src/main.py --nepochs 160 --lr 0.001 --wd 0 --datasets cifar100 --network wide_resnet20 --batch_size 64 --exp-name lwf_fea_widen --approach lwf_fea_CosLR --num-tasks 10 --seed 42
#python src/main.py --nepochs 160 --lr 0.001 --wd 0 --datasets cifar100 --network wide_resnet20 --batch_size 64 --exp-name lwf_fea_widen --approach lwf_fea_CosLR --num-tasks 10 --seed 42
#python src/main.py --nepochs 160 --lr 0.001 --wd 0 --datasets cifar100 --network wide_resnet20 --batch_size 64 --exp-name lwf_fea_widen --approach lwf_fea_CosLR --num-tasks 10 --seed 42
#python src/main.py --nepochs 160 --lr 0.001 --wd 0 --datasets tiny_imagenet_200 --network resnet32 --batch_size 64 --exp-name lwf_fea_3_1_haveBNl_COSLR2_333 --approach lwf_fea_CosLR --num-tasks 25