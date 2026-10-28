import numpy as np
import pytest
import torch
from uci_training import split_indices, ArrayDataset, standardize_inplace, train_one_fold


def test_supplied_split_roles_and_no_leakage():
    test=np.array([[1],[0],[0],[0]])
    validation=np.array([[0],[1],[0],[0]])
    train,val,held=split_indices(test,validation,0)
    assert train.tolist()==[2,3] and val.tolist()==[1] and held.tolist()==[0]
    x=np.array([[100.],[200.],[1.],[3.]])
    stats=standardize_inplace(x,train)
    np.testing.assert_allclose(stats['mean'],[2.])
    ds=ArrayDataset(x,np.array([0,1,0,1]),train)
    assert ds[0][0].shape==(1,) and ds[0][1].shape==(2,)


def test_overlap_rejected():
    with pytest.raises(ValueError,match='overlap'):
        split_indices(np.array([[1],[0],[0]]),np.array([[1],[0],[0]]),0)


def test_one_epoch_training_and_resume(tmp_path):
    rng=np.random.default_rng(13)
    x=rng.normal(size=(30,2));y=(x[:,0]>0).astype(np.int64)
    args=dict(y=y,train_idx=np.arange(18),val_idx=np.arange(18,24),test_idx=np.arange(24,30),
              algorithm='ldlt-resnet',seed=1,log_dir=tmp_path,hparams={'protocol_version':2,'split_policy':'test_mask_one'},
              max_epochs=1,depth=2,batch_size=8,device=torch.device('cpu'))
    result=train_one_fold(X=x.copy(),**args)
    repeated=train_one_fold(X=x.copy(),**args)
    assert result['test_acc']==repeated['test_acc']
    assert result['cert_evals']==repeated['cert_evals']
    assert result['certificate_audit']['min_eigenvalue']>0
