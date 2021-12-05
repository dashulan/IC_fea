for SEED in  1242 4830  5660
do
  python src/main.py --nepochs 160 --lr 0.001 --wd 0  --batch_size 64 \
          --datasets cifar100 --network resnet18 --approach lwf_fea_CosLR \
          --exp-name lwf_fea_${SEED}_init --num-tasks 10 --seed $SEED
done

for SEED in 42 1242 4830 3528 5660
do
  python src/main.py --nepochs 160 --lr 0.001 --wd 0  --batch_size 64 \
          --datasets cifar100 --network resnet18 --approach lwf_fea_CosLR \
          --exp-name lwf_fea_${SEED}_init --num-tasks 20 --seed $SEED
done