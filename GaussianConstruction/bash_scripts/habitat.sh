# train
python scripts\\habitat_splatam.py configs\\habitat\\habitat_splatam.py

# eval
python viz_scripts/final_recon.py configs/habitat/habitat_splatam.py

python viz_scripts/final_recon_sem.py configs/habitat/habitat_splatam.py

python viz_scripts/online_recon.py configs/habitat/habitat_splatam.py

python test\\render_novel_view.py configs\\habitat\\habitat_splatam.py

python -m debugpy --listen 5678 --wait-for-client test\\render_novel_view.py configs\\habitat\\habitat_splatam.py