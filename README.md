# Adversarial Attack for NLP

This is the official repo of our COLING 2020 Paper: **[A Geometry-Inspired Attack for Generating Natural Language Adversarial Examples](https://www.aclweb.org/anthology/2020.coling-main.585.pdf)**.

## Example Usage

### Training
      python main.py --batch_size 500 --model lstm --max_steps 600 --dataset imdb --hidden_size 128 --embedding glove.6B --vocab_size 60000 --embedding_size 100 --num_worker 0

### Adv Training

      python main.py --adv --perturb_correct --n_samples_to_disk -1 --batch_size 250 --max_steps 600 --max_loops 100 --model lstm --dataset imdb --hidden_size 128 --embedding glove.6B --vocab_size 60000 --embedding_size 100 --num_worker 0 --attack deepfool --load_model --model_path experiments/baseline/saved_models/imdb_lstm_esize_100_glove.6B_u_lr_0.001_h_128_bt_1000_s_600/9.pth

### Attack

      python main.py --max_loops 50 --perturb_correct  --n_samples_to_disk -1 --batch_size 250 --model lstm --max_steps 600 --dataset imdb --hidden_size 128 --embedding glove.6B --vocab_size 60000 --embedding_size 100 --num_worker 0 --attack deepfool --load_model --model_path experiments/baseline/saved_models/imdb_lstm_esize_100_glove.6B_u_lr_0.001_h_128_bt_1000_s_600/9.pth

Details of commandline options in models/train.py.

## Citation

```
@inproceedings{meng2020geometry,
  title={A Geometry-Inspired Attack for Generating Natural Language Adversarial Examples},
  author={Meng, Zhao and Wattenhofer, Roger},
  booktitle={Proceedings of the 28th International Conference on Computational Linguistics},
  pages={6679--6689},
  year={2020}
}
```
