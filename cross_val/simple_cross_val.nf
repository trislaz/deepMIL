#!/usr/bin/env nextflow

dataset = 'curie'
params.expe = 'afa_formol'
expe = params.expe
config_path = '/mnt/data4/tlazard/projets/deepMIL/cross_val/handcrafted_configs/config_curie.yaml'
model_name = 'attentionmil'
test_fold = 5 
repetition = 5
epochs = 100
res = 2
visu_pred = 0

// Channels building.
with_test = 1
config_chan = Channel.from("${config_path}") .set{config_chan1; config_chan2}
model_res_conf = model_name_1. combine(res_conf1)

process Train {
    publishDir "${output_folder}", overwrite: true, pattern: "*.pt.tar", mode: 'copy'
    publishDir "${output_folder}", overwrite: true, pattern: "*eventsevents.*", mode: 'copy'

	queue "gpu-cbio"
    clusterOptions "--gres=gpu:1"
    maxForks 10
    memory '40GB'
	cpus 6
 

    input:
    val(config) from config_chan1
    each test from 0..test_fold-1
    each repeat from 1..repetition

    output:
    file('*.pt.tar') into results

    script:
    py = file('../scripts/train.py')
    root_expe = file("./outputs/${dataset}/${expe}/") 
    output_folder = file("${root_expe}/test_${test}/${repeat}/")
    """
    export EVENTS_TF_FOLDER=${output_folder}
	module load cuda10.0
    python $py --config ${config} --test_fold $test --epochs $epochs --repeat $repeat --model_name ${model}
    """
}

results .groupTuple()
        .into {all_done1; all_done2}

process copyconfig {
		
	input: 
	val _ from all_done1
    val(config) from config_chan2

	output:

	script:
    root_expe = file("./outputs/${dataset}/${expe}/") 
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
    val _ from all_done2 

    output:
    file('*.csv') into table_results
    file('*.pt.tar') into best_test_models
    file('model_best_*.pt.tar') into checkpoints

    script:
    root_expe = file("./outputs/${dataset}/${expe}/") 
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
        val _ from best_test_models

        output:
        file('*.csv') into test_results

        script:
        root_expe = file("./outputs/${dataset}/${expe}/") 
        output_folder = root_expe
        py = file('../scripts/writes_final_results.py')
        """
        module load cuda10.0
        python $py --path ${output_folder}
        """
    }
}

if (visu_pred == 1) {
    checkpoints .flatMap()
                .set{input_visu}

     process VisuResults {
        publishDir "${output_folder}/summaries/", overwrite:true, pattern:"*.jpg", mode:'copy'
    	queue "gpu-cbio"
    	clusterOptions "--gres=gpu:1"
    	memory '40GB'
		cpus 7


        input:
        file(model) from input_visu

        output:
        file('*.jpg') into visu_results

        script:
        root_expe = file("./outputs/${dataset}/${expe}/${model}/res_${res}/") 
        output_folder = root_expe
        py = file('../scripts/summary_from_table.py')
        """
        module load cuda10.0
        python $py --model_path ${model} --with_gt 1
        """
    }   
}
