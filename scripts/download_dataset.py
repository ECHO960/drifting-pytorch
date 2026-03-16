from huggingface_hub import snapshot_download

snapshot_download(                                                                                                                                              
    repo_id='ILSVRC/imagenet-1k',                                                                             
    repo_type='dataset',
    local_dir='/mnt/bn/arnold-tiktok-search-raw-rel/chunhui.liu/datasets/imagenet-hf',
    token='',
)
