

### start pretrain
compound_encoder_config="model_configs/geognn_l8.json"
model_config="model_configs/pretrain_gem.json"
dataset="zinc"
data_path="./demo_zinc_smiles"
python pretrain.py \
		--batch_size=256 \
		--num_workers=4 \
		--max_epoch=50 \
		--learning_rate=1e-3 \
		--dropout_rate=0.2 \
		--dataset=$dataset \
		--data_path=$data_path \
		--compound_encoder_config=$compound_encoder_config \
		--model_config=$model_config \
		--model_dir=./pretrain_models/$dataset
