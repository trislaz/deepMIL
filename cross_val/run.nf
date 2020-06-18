#!/usr/bin/env nextflow

model_name = Channel.from('conan', '1s', 'attentionmil')
resolution = Channel.from(1, 2)
model_res = model_name .combine (resolution)
dataset = 'tcga_all'
path_wsi = '/path/'
table_data = '/path/'
nb_para = 2
test_fold = 4
repetition = 15
epochs = 100


//
//
//TODO ADD THE CONFS FOR THE PROCESSES ( queue etc... )
//
//


// This process samples the hyperparameters for all the runs.
// I choose to use a random sampling for the search of parameters.
process SampleHyperparameter {
    publishDir "${output_folder}/configs/", overwrite: true, pattern: "*.yaml", mode: 'copy'
    queue 'cpu'

    input:
    set val(model), val(res) from model_res

    output:
    set val(model), val(res), file('*.yaml') into configs

    script:
    output_folder = "./outputs/${dataset}/${model}/${res}/"
    py = file("./hyperparameter_sampler.py")
    """
    python $py --model_name ${model} \
               --path_wsi ${path_wsi} \
               --table_data ${table_data} \
               --nb_para ${nb_para} \
               --res ${res} 
    """

}

// Trains the models for each (n, t, r)
process Train {
    publishDir "${output_folder}", overwrite: true, pattern: "*.pt.tar", mode: 'copy'

    input:
    val(model), val(res), file(config) from configs
    each test from 1..test_fold
    each repeat from 1..repetition

    output:
    set val(model), file('*.pt.tar') into results

    script:
    py = '/Users/trislaz/Documents/cbio/projets/deepMIL_tris/train/train.py'
    output_folder = "./outputs/${dataset}/${model}/${res}/${config.baseName}/test_${test}/${repeat}/"
    config.copyTo(output_folder)
    
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
    publishDir "${output_folder}", overwrite: true, pattern: "*.yaml", mode: 'copy'
    publishDir "${output_folder}", overwrite: true, pattern: "*.pt.tar", mode: 'copy'

    input:
    set val(model), _ from all_done 
    each res from resolution

    output:
    file('*.csv') into table_results

    script:
    output_folder = "./outputs/${dataset}/${model}/${res}/"
    py = './writes_results.py'
    """
    python $py --path ${output_folder} 
    """
}