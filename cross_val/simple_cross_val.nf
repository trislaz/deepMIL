#!/usr/bin/env nextflow

model_name = Channel.from('conan')
resolution = Channel.from(1, 2) .into{ resolution1; resolution2; resolution3}
model_res = model_name .combine (resolution1)
dataset = 'tcga_all'

// Useful Path
table_data = '/mnt/data4/tlazard/data/tcga/tcga_all/tcga_balanced_lst.csv'
config = file('/mnt/data4/tlazard/projet/deepMIL/config_default.yaml')

// Training parameters
target_name = 'LST_status'
nb_para = 50
test_fold = 5
repetition = 5
epochs = 100

process Train {
    publishDir "${output_folder}", overwrite: true, pattern: "*.pt.tar", mode: 'copy'
    publishDir "${output_folder}", overwrite: true, pattern: "*.yaml", mode: 'copy'

	queue "gpu-cbio"
    clusterOptions "--gres=gpu:1"
    maxForks 10
    memory '30GB'
	cpus 6
 

    input:
    set val(model), val(res) from model_res
    each test from 0..test_fold-1
    each repeat from 1..repetition

    output:
    set val(model), file('*.pt.tar') into results

    script:
    py = file('../train/train.py')
    output_folder = file("./outputs/${dataset}/${model}/${res}/test_${test}/${repeat}/")
    config.copyTo(file("./outputs/${dataset}/${model}/${res}/"))
    """
	module load cuda10.0
    python $py --config ${config} --test_fold $test --epochs $epochs --repeat $repeat
    """
}

results .groupTuple()
        .set {all_done}

// Exctracts the output files, the best parameters and best model.
// needs to write this for each res/model_name. 
// NE PAS chercher à collate tout. Effectuer quand tous les process précédents sont terminer, prend en entrée les tuples différents de ()
process WritesResultFile {
    publishDir "${output_folder}", overwrite: true, pattern: "*.pt.tar", mode: 'copy'
    publishDir "${output_folder}", overwrite: true, pattern: "*.csv", mode:'copy'

    input:
    set val(model), _ from all_done 
    each res from resolution2

    output:
    file('*.csv') into table_results
    set val(model), val("$res"), file('*.pt.tar') into best_test_models
	file("*.yaml")

    script:
    output_folder = file("./outputs/${dataset}/${model}/${res}/")
    py = file('./writes_results.py')
    """
    python $py --path ${output_folder} 
    """
}

process TestResults {
    publishDir "${output_folder}", overwrite:true, pattern:"*.csv", mode:'copy'
	queue "gpu-cbio"
	clusterOptions "--gres=gpu:1"
	memory '30GB'


    input:
    set model, res, _ from best_test_models

    output:
    file('*.csv') into test_results

    script:
    output_folder = file("./outputs/${dataset}/${model}/${res}")
    py = file('./writes_final_results.py')
    """
    module load cuda10.0
    python $py --path ${output_folder}
    """
}
