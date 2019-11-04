python mf_model.py mf_vae_010 --data_mode mean2form --form vae --form_dim 10 --n_epoch 20
python mf_model.py mf_vae_050 --data_mode mean2form --form vae --form_dim 50 --n_epoch 20
python mf_model.py mf_vae_100 --data_mode mean2form --form vae --form_dim 100 --n_epoch 20
python mf_model.py mf_svd_010 --data_mode mean2form --form svd --form_dim 10 --n_epoch 20
python mf_model.py mf_svd_050 --data_mode mean2form --form svd --form_dim 50 --n_epoch 20
python mf_model.py mf_svd_100 --data_mode mean2form --form svd --form_dim 100 --n_epoch 20
REM form2mean
python mf_model.py fm_vae_010 --data_mode form2mean --form vae --form_dim 10 --n_epoch 20
python mf_model.py fm_vae_050 --data_mode form2mean --form vae --form_dim 50 --n_epoch 20
python mf_model.py fm_vae_100 --data_mode form2mean --form vae --form_dim 100 --n_epoch 20
python mf_model.py fm_svd_010 --data_mode form2mean --form svd --form_dim 10 --n_epoch 20
python mf_model.py fm_svd_050 --data_mode form2mean --form svd --form_dim 50 --n_epoch 20
python mf_model.py fm_svd_100 --data_mode form2mean --form svd --form_dim 100 --n_epoch 20