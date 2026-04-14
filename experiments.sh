######################################## EXPERIMENT 1 ########################################
# Diferent initialization in various incremental learning approaches.
##############################################################################################
seeds=(11 22 33 44 55)
datasets=("FBinc-S" "FBinc-M" "FBinc-L" "EventInc")
inits=(1)
RNS=(0 0.1 0.5)
models=("LKGE" "finetune" "incDE" "EWC" "EMR")
lrs=(0.001 0.0005 0.0001)

for seed in "${seeds[@]}"; do
  for lr in "${lrs[@]}"; do
    for dataset in "${datasets[@]}"; do
      for model in "${models[@]}"; do
        for init in "${inits[@]}"; do
          for RN in "${RNS[@]}"; do
            # Run the Python script with the current combination of parameters
            python main.py -dataset "$dataset" -gpu 0 -lifelong_name "$model" -init 1 -RN "$RN" -seed "$seed" -learning_rate "$lr"
          done
        done
      done
    done
  done
done


seeds=(11 22 33 44 55)
datasets=("FBinc-S" "FBinc-M" "FBinc-L" "EventInc")
inits=(0 3)
models=("LKGE" "finetune" "incDE" "EWC" "EMR")
lrs=(0.001 0.0005 0.0001)

for seed in "${seeds[@]}"; do
  for lr in "${lrs[@]}"; do
    for dataset in "${datasets[@]}"; do
      for init in "${inits[@]}"; do
          for model in "${models[@]}"; do
              # Run the Python script with the current combination of parameters
              python main.py -dataset "$dataset" -gpu 0 -lifelong_name "$model" -init "$model" -seed "$seed" -learning_rate "$lr"
          done
      done
    done
  done
done



######################################## EXPERIMENT 2 ########################################
# Number of training epochs with the best results obtained previosuly
##############################################################################################


##### FBinc-S
datasets=("FBinc-S")
epochs=(5 10 15 20 25 30 35 40 45 50 75 100 150 200)
models=("finetune" "EWC" "EMR" "LKGE" "incDE")
inits=(0 1 3)

for dataset in "${datasets[@]}"; do
  for epoch in "${epochs[@]}"; do
    for init in "${inits[@]}"; do
      for model in "${models[@]}"; do
        # Run the Python script with the current combination of parameters
        python main.py -dataset "$dataset" -gpu 0 -lifelong_name "$model" -init "$init" -incremental_epochs "$epoch" -learning_rate 0.0005
      done
    done
  done
done


##### FBinc-M
datasets=("FBinc-M")
epochs=(5 10 15 20 25 30 35 40 45 50 75 100 150 200)
models=("finetune" "EMR")
inits=(0)

for dataset in "${datasets[@]}"; do
  for epoch in "${epochs[@]}"; do
    for init in "${inits[@]}"; do
      for model in "${models[@]}"; do
        # Run the Python script with the current combination of parameters
        python main.py -dataset "$dataset" -gpu 0 -lifelong_name "$model" -init "$init" -incremental_epochs "$epoch" -learning_rate 0.0001
      done
    done
  done
done


datasets=("FBinc-M")
epochs=(5 10 15 20 25 30 35 40 45 50 75 100 150 200)
models=("EWC" "LKGE" "incDE")
inits=(0)

for dataset in "${datasets[@]}"; do
  for epoch in "${epochs[@]}"; do
    for init in "${inits[@]}"; do
      for model in "${models[@]}"; do
        # Run the Python script with the current combination of parameters
        python main.py -dataset "$dataset" -gpu 0 -lifelong_name "$model" -init "$init" -incremental_epochs "$epoch" -learning_rate 0.0005
      done
    done
  done
done

datasets=("FBinc-M")
epochs=(5 10 15 20 25 30 35 40 45 50 75 100 150 200)
models=("finetune" "EWC" "EMR" "LKGE")
inits=(3)

for dataset in "${datasets[@]}"; do
  for epoch in "${epochs[@]}"; do
    for init in "${inits[@]}"; do
      for model in "${models[@]}"; do
        # Run the Python script with the current combination of parameters
        python main.py -dataset "$dataset" -gpu 0 -lifelong_name "$model" -init "$init" -incremental_epochs "$epoch" -learning_rate 0.00005
      done
    done
  done
done

datasets=("FBinc-M")
epochs=(5 10 15 20 25 30 35 40 45 50 75 100 150 200)
models=("finetune" "EWC" "EMR" "incDE")
inits=(3)

for dataset in "${datasets[@]}"; do
  for epoch in "${epochs[@]}"; do
    for init in "${inits[@]}"; do
      for model in "${models[@]}"; do
        # Run the Python script with the current combination of parameters
        python main.py -dataset "$dataset" -gpu 0 -lifelong_name "$model" -init "$init" -incremental_epochs "$epoch" -learning_rate 0.0005
      done
    done
  done
done


datasets=("FBinc-M")
epochs=(5 10 15 20 25 30 35 40 45 50 75 100 150 200)
models=("finetune" "EWC" "EMR" "LKGE")
inits=(1)

for dataset in "${datasets[@]}"; do
  for epoch in "${epochs[@]}"; do
    for init in "${inits[@]}"; do
      for model in "${models[@]}"; do
        # Run the Python script with the current combination of parameters
        python main.py -dataset "$dataset" -gpu 0 -lifelong_name "$model" -init "$init" -incremental_epochs "$epoch" -learning_rate 0.00005
      done
    done
  done
done

datasets=("FBinc-M")
epochs=(5 10 15 20 25 30 35 40 45 50 75 100 150 200)
models=("incDE")
inits=(1)

for dataset in "${datasets[@]}"; do
  for epoch in "${epochs[@]}"; do
    for init in "${inits[@]}"; do
      for model in "${models[@]}"; do
        # Run the Python script with the current combination of parameters
        python main.py -dataset "$dataset" -gpu 0 -lifelong_name "$model" -init "$init" -incremental_epochs "$epoch" -learning_rate 0.0005
      done
    done
  done
done


#### FBinc-L


datasets=("FBinc-L")
epochs=(5 10 15 20 25 30 35 40 45 50 75 100 150 200)
models=("finetune" "EMR")
inits=(0)

for dataset in "${datasets[@]}"; do
  for epoch in "${epochs[@]}"; do
    for init in "${inits[@]}"; do
      for model in "${models[@]}"; do
        # Run the Python script with the current combination of parameters
        python main.py -dataset "$dataset" -gpu 0 -lifelong_name "$model" -init "$init" -incremental_epochs "$epoch" -learning_rate 0.00005
      done
    done
  done
done


datasets=("FBinc-L")
epochs=(5 10 15 20 25 30 35 40 45 50 75 100 150 200)
models=("EWC" "LKGE" "incDE")
inits=(0)

for dataset in "${datasets[@]}"; do
  for epoch in "${epochs[@]}"; do
    for init in "${inits[@]}"; do
      for model in "${models[@]}"; do
        # Run the Python script with the current combination of parameters
        python main.py -dataset "$dataset" -gpu 0 -lifelong_name "$model" -init "$init" -incremental_epochs "$epoch" -learning_rate 0.0005
      done
    done
  done
done


datasets=("FBinc-L")
epochs=(5 10 15 20 25 30 35 40 45 50 75 100 150 200)
models=("finetune" "EMR")
inits=(3)

for dataset in "${datasets[@]}"; do
  for epoch in "${epochs[@]}"; do
    for init in "${inits[@]}"; do
      for model in "${models[@]}"; do
        # Run the Python script with the current combination of parameters
        python main.py -dataset "$dataset" -gpu 0 -lifelong_name "$model" -init "$init" -incremental_epochs "$epoch" -learning_rate 0.00005
      done
    done
  done
done


datasets=("FBinc-L")
epochs=(5 10 15 20 25 30 35 40 45 50 75 100 150 200)
models=("EWC" "LKGE" "incDE")
inits=(3)

for dataset in "${datasets[@]}"; do
  for epoch in "${epochs[@]}"; do
    for init in "${inits[@]}"; do
      for model in "${models[@]}"; do
        # Run the Python script with the current combination of parameters
        python main.py -dataset "$dataset" -gpu 0 -lifelong_name "$model" -init "$init" -incremental_epochs "$epoch" -learning_rate 0.0005
      done
    done
  done
done



datasets=("FBinc-L")
epochs=(5 10 15 20 25 30 35 40 45 50 75 100 150 200)
models=("finetune" "EMR")
inits=(1)

for dataset in "${datasets[@]}"; do
  for epoch in "${epochs[@]}"; do
    for init in "${inits[@]}"; do
      for model in "${models[@]}"; do
        # Run the Python script with the current combination of parameters
        python main.py -dataset "$dataset" -gpu 0 -lifelong_name "$model" -init "$init" -incremental_epochs "$epoch" -learning_rate 0.00005
      done
    done
  done
done


datasets=("FBinc-L")
epochs=(5 10 15 20 25 30 35 40 45 50 75 100 150 200)
models=("EWC")
inits=(1)

for dataset in "${datasets[@]}"; do
  for epoch in "${epochs[@]}"; do
    for init in "${inits[@]}"; do
      for model in "${models[@]}"; do
        # Run the Python script with the current combination of parameters
        python main.py -dataset "$dataset" -gpu 0 -lifelong_name "$model" -init "$init" -incremental_epochs "$epoch" -learning_rate 0.0005 -RN 0.1
      done
    done
  done
done


datasets=("FBinc-L")
epochs=(5 10 15 20 25 30 35 40 45 50 75 100 150 200)
models=("LKGE")
inits=(1)

for dataset in "${datasets[@]}"; do
  for epoch in "${epochs[@]}"; do
    for init in "${inits[@]}"; do
      for model in "${models[@]}"; do
        # Run the Python script with the current combination of parameters
        python main.py -dataset "$dataset" -gpu 0 -lifelong_name "$model" -init "$init" -incremental_epochs "$epoch" -learning_rate 0.0005 -RN 0.5
      done
    done
  done
done


datasets=("FBinc-L")
epochs=(5 10 15 20 25 30 35 40 45 50 75 100 150 200)
models=("incDE")
inits=(1)

for dataset in "${datasets[@]}"; do
  for epoch in "${epochs[@]}"; do
    for init in "${inits[@]}"; do
      for model in "${models[@]}"; do
        # Run the Python script with the current combination of parameters
        python main.py -dataset "$dataset" -gpu 0 -lifelong_name "$model" -init "$init" -incremental_epochs "$epoch" -learning_rate 0.0005 -RN 0.1s
      done
    done
  done
done


#### EventInc


datasets=("EventInc")
epochs=(5 10 15 20 25 30 35 40 45 50 75 100 150 200)
models=("finetune")
inits=(0 3)

for dataset in "${datasets[@]}"; do
  for epoch in "${epochs[@]}"; do
    for init in "${inits[@]}"; do
      for model in "${models[@]}"; do
        # Run the Python script with the current combination of parameters
        python main.py -dataset "$dataset" -gpu 0 -lifelong_name "$model" -init "$init" -incremental_epochs "$epoch" -learning_rate 0.00005
      done
    done
  done
done

datasets=("EventInc")
epochs=(5 10 15 20 25 30 35 40 45 50 75 100 150 200)
models=("finetune")
inits=(1)

for dataset in "${datasets[@]}"; do
  for epoch in "${epochs[@]}"; do
    for init in "${inits[@]}"; do
      for model in "${models[@]}"; do
        # Run the Python script with the current combination of parameters
        python main.py -dataset "$dataset" -gpu 0 -lifelong_name "$model" -init "$init" -incremental_epochs "$epoch" -learning_rate 0.0001 -RN 0.5
      done
    done
  done
done


datasets=("EventInc")
epochs=(5 10 15 20 25 30 35 40 45 50 75 100 150 200)
models=("EMR")
inits=(0 1)

for dataset in "${datasets[@]}"; do
  for epoch in "${epochs[@]}"; do
    for init in "${inits[@]}"; do
      for model in "${models[@]}"; do
        # Run the Python script with the current combination of parameters
        python main.py -dataset "$dataset" -gpu 0 -lifelong_name "$model" -init "$init" -incremental_epochs "$epoch" -learning_rate 0.0001 -RN 0.1
      done
    done
  done
done

datasets=("EventInc")
epochs=(5 10 15 20 25 30 35 40 45 50 75 100 150 200)
models=("EMR" "EWC")
inits=(3)

for dataset in "${datasets[@]}"; do
  for epoch in "${epochs[@]}"; do
    for init in "${inits[@]}"; do
      for model in "${models[@]}"; do
        # Run the Python script with the current combination of parameters
        python main.py -dataset "$dataset" -gpu 0 -lifelong_name "$model" -init "$init" -incremental_epochs "$epoch" -learning_rate 0.00005
      done
    done
  done
done


datasets=("EventInc")
epochs=(5 10 15 20 25 30 35 40 45 50 75 100 150 200)
models=("EWC" "incDE")
inits=(0 1)

for dataset in "${datasets[@]}"; do
  for epoch in "${epochs[@]}"; do
    for init in "${inits[@]}"; do
      for model in "${models[@]}"; do
        # Run the Python script with the current combination of parameters
        python main.py -dataset "$dataset" -gpu 0 -lifelong_name "$model" -init "$init" -incremental_epochs "$epoch" -learning_rate 0.0001 -RN 0.5
      done
    done
  done
done


datasets=("EventInc")
epochs=(5 10 15 20 25 30 35 40 45 50 75 100 150 200)
models=("LKGE")
inits=(0 1 3)

for dataset in "${datasets[@]}"; do
  for epoch in "${epochs[@]}"; do
    for init in "${inits[@]}"; do
      for model in "${models[@]}"; do
        # Run the Python script with the current combination of parameters
        python main.py -dataset "$dataset" -gpu 0 -lifelong_name "$model" -init "$init" -incremental_epochs "$epoch" -learning_rate 0.0001 -RN 0.5
      done
    done
  done
done



datasets=("EventInc")
epochs=(5 10 15 20 25 30 35 40 45 50 75 100 150 200)
models=("incDE")
inits=(1)

for dataset in "${datasets[@]}"; do
  for epoch in "${epochs[@]}"; do
    for init in "${inits[@]}"; do
      for model in "${models[@]}"; do
        # Run the Python script with the current combination of parameters
        python main.py -dataset "$dataset" -gpu 0 -lifelong_name "$model" -init "$init" -incremental_epochs "$epoch" -learning_rate 0.0005
      done
    done
  done
done