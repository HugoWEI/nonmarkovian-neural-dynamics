"""Train/evaluate neural residual closure against classical interpolation.

NEVER train on held-out higher-depth trajectories. One network learns the
residual beyond first-order Schur/Volterra feedback; a matched network
learns high-minus-low correction directly. Ordinary tensor-product linear
interpolation of residuals is a strong *non-neural* baseline.

The purpose is falsification, not to claim a new solver or speedup.
"""
import json,time
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator


def feature_tensor(params,time,low,first):
    N,T=low.shape
    tt=np.broadcast_to(time[None,:]/12.,(N,T))
    return np.stack([
        tt,
        np.broadcast_to(((params[:,0]-1.)/.3)[:,None],(N,T)),
        np.broadcast_to(((params[:,1]-.1)/.1)[:,None],(N,T)),
        low,
        first-low
    ],axis=-1)


def train_net(data,mode,seed,steps=2500):
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    dtype=torch.float64
    idx=np.where(data["split"]=="train")[0]
    features=feature_tensor(data["params"][idx],data["time"],
                            data["low"][idx],data["first"][idx])
    y=(data["high"][idx]-(
        data["first"][idx] if mode=="residual" else data["low"][idx]))
    input_tensor=torch.tensor(features,dtype=dtype)
    target=torch.tensor(y/.01,dtype=dtype)
    t=torch.tensor((data["time"]/12.)[None,:],dtype=dtype)
    power=4 if mode=="residual" else 2
    model=torch.nn.Sequential(
        torch.nn.Linear(5,64),torch.nn.Tanh(),
        torch.nn.Linear(64,64),torch.nn.Tanh(),
        torch.nn.Linear(64,1)
    ).to(dtype=dtype)
    optim=torch.optim.Adam(model.parameters(),lr=.003)
    weights=1.+3.*t**4
    started=time.perf_counter()
    for _ in range(steps):
        optim.zero_grad(set_to_none=True)
        pred=t**power*model(input_tensor).squeeze(-1)
        loss=(weights*(pred-target).square()).mean()
        loss+=3.*(pred[:,-1]-target[:,-1]).square().mean()
        loss.backward();optim.step()
    train_s=time.perf_counter()-started
    all_features=torch.tensor(feature_tensor(
        data["params"],data["time"],data["low"],data["first"]),dtype=dtype)
    with torch.no_grad():
        residual=.01*((all_features[...,0]**power)*
             model(all_features).squeeze(-1)).numpy()
    predicted=(data["first"] if mode=="residual" else data["low"])+residual
    return predicted,train_s


def metric(pred,actual):
    diff=np.asarray(pred)-np.asarray(actual)
    return {
        "terminal_abs_error":float(np.mean(abs(diff[:,-1]))),
        "trajectory_rmse":float(np.sqrt(np.mean(diff**2))),
        "max_abs_error":float(max(abs(diff.ravel()))),
        "outside_unit_interval_max":float(max(
            max(0.,-np.min(pred)),max(0.,np.max(pred)-1.)))
    }


def run(dataset_path="outputs/neural_memory_dataset/trajectories.npz",
        output="outputs/neural_memory_closure"):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    with np.load(dataset_path) as z:
        data={k:z[k] for k in z.files}
    idx_train=np.where(data["split"]=="train")[0]
    # The training pairs form an exact tensor-product parameter grid.
    u_grid=np.array([.8,1.,1.2])
    mu_grid=np.array([.025,.075,.125,.2])
    true_training_residual=(data["high"][idx_train]-data["first"][idx_train])
    grid_values=true_training_residual.reshape(3,4,-1)
    interp=RegularGridInterpolator(
        (u_grid,mu_grid),grid_values,method="linear",
        bounds_error=False,fill_value=None)
    interp_result=data["first"]+interp(data["params"])
    predictions={"low":data["low"],"first_order":data["first"],
                 "linear_residual_interpolation":interp_result}
    train_time={}
    for seed in (1,7):
        for mode in ("residual","direct"):
            pred,seconds=train_net(data,mode,seed)
            predictions[f"neural_{mode}_seed{seed}"]=pred
            train_time[f"neural_{mode}_seed{seed}"]=seconds
            print(json.dumps({"trained_model":mode,"seed":seed,
                              "training_seconds":seconds}),flush=True)
    result=[]
    for split in ("heldout","extrapolate"):
        mask=(data["split"]==split)
        for name,pred in predictions.items():
            record={"split":split,"method":name,"n_cases":int(sum(mask)),
                    **metric(pred[mask],data["high"][mask]),
                    "model_train_seconds":train_time.get(name,0.)}
            result.append(record)
            print(json.dumps(record),flush=True)
    (out/"results.json").write_text(json.dumps(result,indent=2))
    lines=[
        "# Interacting 12ps neural memory-closure: held-out tests",
        "",
        "**Fixed low-T Nk7, interacting U>0, one-lead Anderson HEOM.**",
        "All high-depth teacher states share a projected depth2-correlated initial",
        "state; omitted tier ADOs are zero. This is not source full equilibrium.",
        "The higher-depth trajectories at held-out/extrapolated parameters",
        "were not used to train neural models or the interpolation baseline.",
        "Training: 12 physical parameter pairs × 161 time points.",
        "Testing: four interpolating held-out pairs and two extrapolated pairs.",
        "Matched 64×64 tanh MLP, 2500 Adam steps, seeds 1 and 7.",
        "Strong classical baseline: tensor-product linear interpolation of",
        "the **higher-order residual** on the training parameter grid.",
        "",
        "| split | model | final abs error | trajectory RMSE | train s |",
        "|---|---|---:|---:|---:|",
    ]
    for r in result:
        lines.append(f"| {r['split']} | {r['method']} | "
            f"{r['terminal_abs_error']:.5e} | {r['trajectory_rmse']:.5e} | "
            f"{r['model_train_seconds']:.2f} |")
    lines+=["",
        "A better held-out error alone is not a faster quantum solver.",
        "Assess full high-depth teacher cost and amortization break-even.",
        "A learned closure must outperform interpolation, classical",
        "Volterra correction and direct HEOM under matched physics."]
    (out/"SUMMARY.md").write_text("\n".join(lines)+"\n")
    return result


if __name__=="__main__":
    run()
