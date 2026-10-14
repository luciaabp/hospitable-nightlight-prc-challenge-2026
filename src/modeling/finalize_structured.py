"""Fit final structured bundles reusing verified causal OOF predictions."""
import argparse
import hashlib
import json
import logging
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.feature_engineering.record_quality import (VISIBLE_COLUMNS, record_quality_features,
                                                     record_quality_flags)
from .config import ExperimentConfig
from .data import ID_COLUMN, load_dataset
from .run_structured import VARIANTS, fit_base, fit_gate, predict_gate, rmse
from .structured_mixture import day_shift_mask
from .tracking import environment_info


def finalize(experiment, output, devices, variants=None):
    source, destination = Path(experiment), Path(output)
    if destination.exists():
        raise FileExistsError(destination)
    variants = list(VARIANTS) if variants is None else variants
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")
    settings = json.loads((source/'config.json').read_text())
    config_dict = settings['base_config'].copy()
    config_dict.pop('features', None)
    for field in ['interim_dir', 'experiments_dir']:
        config_dict[field] = Path(config_dict[field])
    config = ExperimentConfig(**config_dict)
    dataset = load_dataset(config)
    X, y, meta = dataset.X, dataset.y.to_numpy(), dataset.meta
    files = sorted(config.interim_dir.glob(config.file_pattern))
    # Old experiments lack content hashes. Refuse reuse if inputs changed since creation.
    cutoff = datetime.strptime(source.name[:15], '%Y%m%d-%H%M%S').replace(tzinfo=timezone.utc).timestamp()
    if any(p.stat().st_mtime > cutoff for p in files):
        raise ValueError('Training files changed since OOF generation; rerun cross-fitting')
    base = pd.read_parquet(source/'base_predictions.parquet')
    pd.testing.assert_frame_equal(meta, base[meta.columns], check_dtype=False)
    np.testing.assert_allclose(base.delay, X.takeoff_minus_SCHED_TIME_UTC_mvt, equal_nan=True)
    previous = pd.read_parquet(source/'predictions.parquet').set_index(ID_COLUMN)
    np.testing.assert_allclose(y[meta[ID_COLUMN].isin(previous.index)], previous.loc[meta.loc[meta[ID_COLUMN].isin(previous.index),ID_COLUMN],'y_true'])
    months = meta.month.to_numpy()
    inner = np.flatnonzero(base.taxi.notna().to_numpy())
    if set(months[inner]) != set(sorted(set(months))[2:]):
        raise ValueError('OOF coverage does not include every month after initial two-month history')
    if not np.isfinite(base.loc[inner,['taxi','clean_taxi','probability']].to_numpy()).all():
        raise ValueError('Incomplete OOF components')
    # Check current feature preparation against the last saved base model too.
    last = max(months)
    models = joblib.load(source/f'models/{last}_base.joblib')
    rows = np.flatnonzero(months == last)
    for name in ['taxi','clean_taxi']:
        np.testing.assert_allclose(models[name].predict(X.iloc[rows]), base.loc[rows,name], atol=1e-8, rtol=1e-12)
    np.testing.assert_allclose(models['classifier'].predict_proba(X.iloc[rows])[:,1],base.loc[rows,'probability'],atol=1e-10)
    del models
    visible = list(dict.fromkeys(VISIBLE_COLUMNS+[ID_COLUMN]))
    raw = pd.concat([pd.read_parquet(p,columns=visible) for p in files],ignore_index=True)
    quality = record_quality_features(raw)
    quality.index = raw[ID_COLUMN]
    quality = quality.loc[meta[ID_COLUMN]].reset_index(drop=True)
    flags = record_quality_flags(raw)
    flags.index = raw[ID_COLUMN]
    flags = flags.loc[meta[ID_COLUMN]].reset_index(drop=True)
    imputed = meta.block_sched_gap_sec.le(60).to_numpy()
    shifted = day_shift_mask(y,imputed)
    params = json.loads((source/'params.json').read_text())
    params['devices'] = devices
    destination.mkdir(parents=True)
    (destination/'models').mkdir()
    settings['base_config']['fit_final_model'] = True
    settings['base_config']['devices'] = devices
    settings.update(fit_final=True, source_experiment=str(source.resolve()), variants=variants)
    (destination/'config.json').write_text(json.dumps(settings,indent=2))
    (destination/'params.json').write_text(json.dumps(params,indent=2))
    (destination/'environment.json').write_text(json.dumps(environment_info(),indent=2))
    manifest = {'status':'training','source_experiment':str(source.resolve()),'training_rows':len(y),
                'training_months':sorted(set(months)), 'oof_rows':len(inner), 'seed':config.seed,
                'variants':variants,
                'data_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                'source_oof_sha256':hashlib.sha256((source/'base_predictions.parquet').read_bytes()).hexdigest()}
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2))
    logging.info('Training final base: %d rows',len(y))
    models,_ = fit_base(X,y,imputed,shifted,np.arange(len(y)),np.array([0]),params,config.categorical_features)
    joblib.dump(models,destination/'models/final_base.joblib')
    tuning_month = max(months[inner])
    fitting, tuning = inner[months[inner]<tuning_month], inner[months[inner]==tuning_month]
    selection = []
    for name in variants:
        spec = VARIANTS[name]
        scores = []
        prior = max(1e-7,float(shifted[months<tuning_month].mean()))
        for penalty in settings['penalties']:
            gate = fit_gate(X,quality,flags,base,y,fitting,spec,penalty,prior,shifted)
            scores.append((rmse(y[tuning],predict_gate(gate,X,quality,flags,base,tuning,spec)),penalty))
        _,penalty = min(scores)
        gate = fit_gate(X,quality,flags,base,y,inner,spec,penalty,max(1e-7,float(shifted.mean())),shifted)
        if not gate.optimization['success']:
            raise RuntimeError(f'{name}: {gate.optimization}')
        joblib.dump(gate,destination/f'models/final_{name}.joblib')
        selection.append({'variant':name,'penalty':penalty,'tuning_month':tuning_month,
                          'tuning_scores':scores,'optimization':gate.optimization})
        logging.info('%s penalty=%s',name,penalty)
    (destination/'selection.json').write_text(json.dumps(selection,indent=2))
    manifest['status']='complete'
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return destination


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--devices',default='0')
    p.add_argument('--variants',nargs='+',choices=VARIANTS)
    args=p.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    destination = finalize(args.experiment,args.output,args.devices,args.variants)
    (destination/'command.txt').write_text(shlex.join(sys.orig_argv) + '\n')


if __name__=='__main__':
    main()
