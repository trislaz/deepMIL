#!/usr/bin/env nextflow

model_name = Channel.from('attentionmil') .into{model_name_1; model_name_2}
resolution = Channel.from(1) .into{ resolution1; resolution2; resolution3}
dataset = 'tcga_all'
expe = 'random_pred'
with_test = 1
// PROCESS GIVE ME COMPLIMENTSS {input: give ME, output: SCHNITZELL}
// Useful Path
config_chan = Channel.from('/mnt/data4/tlazard/projets/deepMIL/cross_val/handcrafted_configs/best_config_38.yaml')

res_conf = resolution1 .merge (config_chan) .into{res_conf1; res_conf2}
model_res_conf = model_name_1. combine(res_conf1)
// Training parameters
test_fold = 5 
repetition = 20 
epochs = 100

process Train {
    publishDir "${output_folder}", overwrite: true, pattern: "*.pt.tar", mode: 'copy'
    publishDir "${output_folder}", overwrite: true, pattern: "*eventsevents.*", mode: 'copy'

	queue "gpu-cbio"
    clusterOptions "--gres=gpu:1"
    maxForks 10
    memory '40GB'
	cpus 6
 

    input:
    set val(model), val(res), val(config) from model_res_conf
    each test from 0..test_fold-3
    each repeat from 1..repetition

    output:
    set val(model), file('*.pt.tar') into results

    script:
    py = file('../scripts/train.py')
    root_expe = file("./outputs/${dataset}/${expe}/${model}/res_${res}/") 
    output_folder = file("${root_expe}/test_${test}/${repeat}/")
    """
    export EVENTS_TF_FOLDER=${output_folder}
	module load cuda10.0
    python $py --config ${config} --test_fold $test --epochs $epochs --repeat $repeat
    """
}

results .groupTuple()
        .into {all_done1; all_done2}

process copyconfig {
	input: 
	val _ from all_done1
    set val(res), val(config) from res_conf2
	each model from model_name_2

	output:

	script:
    root_expe = file("./outputs/${dataset}/${expe}/${model}/res_${res}/") 
	output_folder = root_expe
	"""
	cp ${config} ${output_folder}
	"""
}

// Exctracts the output files, the best parameters and best model.
// needs to write this for each res/model_name. 
// NE PAS chercher à collate tout. Effectuer quand tous les process précédents sont terminer, prend en entrée les tuples différents de ()
process WritesResultFile {
    publishDir "${output_folder}", overwrite: true, pattern: "*.pt.tar", mode: 'copy'
    publishDir "${output_folder}", overwrite: true, pattern: "*.csv", mode:'copy'

    input:
    set val(model), _ from all_done2 
    each res from resolution2

    output:
    file('*.csv') into table_results
    set val(model), val("$res"), file('*.pt.tar') into best_test_models

    script:
    root_expe = file("./outputs/${dataset}/${expe}/${model}/res_${res}/") 
    output_folder = root_expe
    py = file('../scripts/writes_results_cross_val.py')
    """
    python $py --path ${output_folder} 
    """
}

if (with_test == 1){
    process TestResults {
        publishDir "${output_folder}", overwrite:true, pattern:"*.csv", mode:'copy'
    	queue "gpu-cbio"
    	clusterOptions "--gres=gpu:1"
    	memory '40GB'
		cpus 7


        input:
        set model, res, _ from best_test_models

        output:
        file('*.csv') into test_results

        script:
        root_expe = file("./outputs/${dataset}/${expe}/${model}/res_${res}/") 
        output_folder = root_expe
        py = file('../scripts/writes_final_results.py')
        """
        module load cuda10.0
        python $py --path ${output_folder}
        """
    }
}
