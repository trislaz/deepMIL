#!/usr/bin/env nextflow

model_name = 'attentionmil'
dataset = 'tcga_all'
res = 1
root_folder = file("./outputs/${dataset}/${model_name}/${res}")
configs = Channel.fromPath("${root_folder}/configs/*.yaml")

process TestResults {
    publishDir "${output_folder}", overwrite:true, pattern:"*.csv", mode:'copy'
	queue "gpu-cbio"
	clusterOptions "--gres=gpu:1"
	memory '40GB'
    maxForks 10
	cpus 6


    input:
    file(config) from configs 

    output:
    file('*.csv') into test_results

    script:
    c = config.baseName
    output_folder = file("${root_folder}/${c}")
    py = file('../scripts/writes_all_final_results.py')
    """
    module load cuda10.0
    python $py --path ${output_folder} --config ${config}
    """
}
