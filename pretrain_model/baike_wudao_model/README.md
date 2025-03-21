## corpus 
pretrain with ***baidubaike/wiki/wudaocorpus_zh***
## split model  
`split -b 99M weights.ckpt weights_part_`
## merge model 
*cat weights_part_\* > weights.ckpt*
## md5check  
*md5sum cfg.ckpt*    
*bf2216457f9c54de7a4dd4e84fd9f3e7  cfg.ckpt*  
*md5sum weights.ckpt*  
*82203e7d56f758634e804a31c5f35a4e  weights.ckpt*  

