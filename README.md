Train a model:

python train.py --model_dir="model/my_model

parameters are like:

{
  "allow_soft_placement": true, 
  "attention_type": "bahdanau", 
  "attn_input_feeding": true, 
  "batch_size": 20, 
  "cell_type": "lstm", 
  "data_balance": true, 
  "decoder_embedding_size": 10, 
  "depth": 1, 
  "display_freq": 4, 
  "dropout_rate": 0.2, 
  "embedding_size": 104, 
  "feature_seq_c_size": 96, 
  "feature_seq_size": 16, 
  "feature_size": 3, 
  "hidden_units": 256, 
  "learning_rate": 0.0001, 
  "log_device_placement": false, 
  "max_epochs": 30, 
  "max_gradient_norm": 1.0, 
  "model_dir": "model/test", 
  "model_name": "translate.ckpt", 
  "noise": 0, 
  "num_decoder_symbols": 3, 
  "num_encoder_symbols": 5000, 
  "optimizer": "adam", 
  "save_freq": 50000, 
  "shuffle_each_epoch": true, 
  "sort_by_length": true, 
  "source_train_data": "session_data/webis-smc-12.data", 
  "source_valid_data": "session_data/webis-smc-12.data", 
  "test_set_partition": 0.05, 
  "time_step": 10, 
  "use_attention": true, 
  "use_dropout": false, 
  "use_fp16": false, 
  "use_residual": false, 
  "valid_freq": 1000000
}



Use a trained model to predict:

python decode.py --model_path="model/my_model/translate.ckpt-xxxx"

xxxx depends on training epochs in the setting.
