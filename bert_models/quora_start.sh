
python sentence_classifier.py \
  --do_train \
	--do_eval \
  --do_lower_case \
  --data_dir /home/panxie/Document/kaggle/quora/data/splited_data \
  --bert_model /home/panxie/Document/GLUE/pre_trained_models/bert-base-uncased.tar.gz \
  --max_seq_length 30 \
  --train_batch_size 64 \
  --learning_rate 5e-5 \
  --num_train_epochs 5.0 \
  --output_dir ./output/
