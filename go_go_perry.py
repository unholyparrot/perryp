__version__ = "0.1.1"

import argparse
import re
import os
# import gzip  # for future parsing
import sys
# from urllib.parse import unquote as un_qt  # possibly for better logging
from itertools import compress

import requests
import yaml
import numpy as np
import pandas as pd

from loguru import logger
from tqdm.auto import tqdm
# from Bio.SeqIO import FastaIO  # for future parsing
from pandarallel import pandarallel
from sklearn.feature_extraction.text import CountVectorizer


def setup_args():
    parser = argparse.ArgumentParser(description="Perry the Platypus — the finder of taxonomy peptide signatures",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-target', '--target',
                        type=int, required=True,
                        help="The taxonId of the highest taxonomy point for the signature search")

    parser.add_argument('-out', '--out_pattern',
                        type=str, required=True,
                        help="Pattern for the output files, should include path but not the extension")

    parser.add_argument('-config', '--config',
                        type=str, default=os.path.join(os.path.split(__file__)[0], "cfg.yaml"),
                        help="Path for the file with configs")

    parser.add_argument('-nw', '--num_workers',
                        type=int, default=10,
                        help="Number of workers for parallelized parts")

    parser.add_argument('-v', '--version', action='version', version=__version__)

    return parser.parse_args()


def pull_sequences_json(aim_taxon_id):
    get_fasta = requests.get("https://rest.uniprot.org/uniprotkb/stream",
                             params={
                                 "format": "json",
                                 "query": f"((taxonomy_id:{aim_taxon_id}))",
                                 "fields": "organism_id,sequence"
                             }
                             )
    if get_fasta.ok:
        response = get_fasta.json()['results']
    else:
        response = None
    return response


def pull_sequences_fasta_gz(aim_taxon_id, proposed_path):
    file_name = proposed_path + f"_sequences_{aim_taxon_id}.fasta.gz"

    with open(file_name, 'wb') as wr_file:
        with requests.get("https://rest.uniprot.org/uniprotkb/stream",
                          params={
                              "compressed": True,
                              "download": True,
                              "format": "fasta",
                              "query": f"((taxonomy_id:{aim_taxon_id}))"
                          },
                          stream=True
                          ) as r:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    wr_file.write(chunk)

    return file_name


def pull_sequences_pagination(aim_taxon_id):
    get_fasta = aim_taxon_id

    if get_fasta.ok:
        response = get_fasta.json()['results']
    else:
        response = None

    return response


def request_taxa(aim_taxon_id, lineage=False):
    get_info = requests.get("https://rest.uniprot.org/taxonomy/stream",
                            params={
                                "fields": "id,parent,rank,scientific_name,lineage" if lineage
                                else "id,parent,rank,scientific_name",
                                "query": f"{aim_taxon_id}"
                            }
                            )
    if get_info.ok:
        response = get_info.json()['results'][0]
    else:
        response = None
    return response


if __name__ == '__main__':
    # читаем и обрабатываем аргументы
    args = setup_args()
    # инициализируем запись логов
    logger.add(args.out_pattern + "_records.log")

    logger.info("Started")
    logger.debug(f"Input taxonId: {args.target}")
    target_id = args.target
    # инициализируем работу параллельного доступа к pandas
    logger.debug(f"Perry will use {args.num_workers} where possible")
    pandarallel.initialize(nb_workers=args.num_workers)

    logger.debug(f"Config path: {args.config}")
    with open(args.config, 'r') as fr:
        config = yaml.load(fr, Loader=yaml.SafeLoader)

    target_info = request_taxa(target_id)
    logger.debug(f"Target taxonId {target_id} seems to be {target_info['scientificName']}; {target_info['rank']} rank")

    logger.debug("Trying to pull all the sequences for this taxonId")

    # максимальная кочерга на запрос последовательностей из UniProt
    sequences_method = 0
    try:
        sequences = pull_sequences_json(target_id)
    except (requests.RequestException, Exception):
        logger.error("Failed pulling sequences as JSON")
        try:
            logger.warning("Trying once again")
            sequences = pull_sequences_json(target_id)
        except (requests.RequestException, Exception):
            logger.error("Twice failed pulling sequences as JSON")
            try:
                logger.warning("I will try to pull FASTA.GZ")
                sequences = pull_sequences_fasta_gz(target_id, args.out_pattern)
            except (requests.RequestException, Exception):
                logger.error("Failed pulling sequences as FASTA.GZ")
                try:
                    logger.warning("I will try to pull by pagination, Go get some tea, that's a really long one")
                    sequences = pull_sequences_pagination(target_id)
                except (Exception, ):
                    logger.critical("Did not succeed at pulling sequences at all. Try again later!")
                    logger.info("Abnormal exit")
                    sys.exit()
                else:
                    sequences_method = 3
            else:
                sequences_method = 2
        else:
            sequences_method = 1
    else:
        sequences_method = 1

    logger.debug("Parsing the sequences")

    records_dict = dict()  # словарь для записи (taxonId : {peptides})
    at_least_all_peps = set()  # множество для записи вообще всех встречаемых пептидов, нужно для создания модели

    # раздел обработки последовательностей
    if sequences_method == 1:
        logger.debug("Most convenient sequences parsing")
        for elem in tqdm(sequences, desc="Parsing the protein sequences"):
            peptides_list = re.sub(r"([KR])([ABCDEFGHIKLMNQRSTVWXYZ])", r"\1\n\2", elem['sequence']['value']).split(
                "\n")
            current_peptides_set = set()
            for pep in peptides_list:
                if config['peptide_len_lb'] <= len(pep) <= config['peptide_len_hb']:
                    current_peptides_set.add(pep)
            if len(current_peptides_set) > 0:
                at_least_all_peps.update(current_peptides_set)
                if records_dict.get(elem['organism']['taxonId']):
                    records_dict[elem['organism']['taxonId']].update(current_peptides_set)
                else:
                    records_dict[elem['organism']['taxonId']] = current_peptides_set
    elif sequences_method == 2:
        logger.error("Oops, not implemented yet")
        logger.info("Abnormal exit")
        sys.exit()
    elif sequences_method == 3:
        logger.error("Oops, not implemented yet")
        logger.info("Abnormal exit")
        sys.exit()
    else:
        logger.error("Oops, not implemented yet")
        logger.info("Abnormal exit")
        sys.exit()

    logger.info(f"Obtained {len(at_least_all_peps)} peptides in total from {len(records_dict)} taxon ids")

    logger.info(f"Some of the taxon ids: {list(records_dict.keys())[:10]}")

    logger.debug("Preparing to pull taxonomy info")
    touched_set = {(target_id, target_info['rank'], target_info['scientificName'], 0)}

    # для записей "подчищенных" по таксономии последовательностей
    filtered_records_dict = dict()

    # TODO: главный кандидат на ускорение. По-хорошему следует передать их в ThreadPoolExecutor, но как-то потом
    for elem in tqdm(list(records_dict.keys()), desc="Pulling taxonomy"):
        filtering_pass = False
        # запрашиваем информацию о текущем таксоне
        c_request = request_taxa(elem, lineage=True)
        # если запрос прошел, то обрабатываем его
        if c_request:
            # создаем искусственно дополненный массив выравнивания
            fulfilled_lineage = [
                                     {
                                         'rank': c_request['rank'],
                                         'scientificName': c_request['scientificName'],
                                         'taxonId': c_request['taxonId']
                                     }
                                 ] + c_request['lineage']

            # если ранг сразу "вид", то мы добавляем его записи в очищенный словарь
            if any(rank_indexer := [x['rank'] == 'species' for x in fulfilled_lineage]):
                # то мы, подразумевая, что такая запись единственная, достаем её и
                # перезаписываем все пептиды как пептиды вида
                parent_species = list(compress(fulfilled_lineage, rank_indexer))[0]['taxonId']
                # может быть такая ситуация, что такой вид уже был добавлен в очищенный словарь,
                # мы не хотим терять пептиды
                # дополняем множество пептидов в таком случае
                if filtered_records_dict.get(parent_species):
                    filtered_records_dict[parent_species].update(records_dict[elem])
                # но если вид встретился впервые, то просто перезаписываем пептиды из "грязного" словаря в "чистый"
                else:
                    filtered_records_dict[parent_species] = records_dict[elem]
                filtering_pass = True
            # возможна и ситуация, что в "грязном" словаре встретились пептиды, которые относятся к таксону ВЫШЕ вида,
            # такие таксоны следует пропускать, но лучше мы уведомим об этом
            else:
                logger.warning(f"Not found 'species' rank any higher than current {elem} taxonId")

            # если успешно прошли фильтрацию принадлежности пептидов, приступаем к выстраиванию таксономии
            if filtering_pass:
                # мы должны сперва отфильтровать ряд происхождения
                # выбираем лишь те из записей в ряду, которые есть среди именованых
                mask_lineage = [x['rank'] in config['ORDERED_RANKING'] for x in fulfilled_lineage]
                # применяем маску к ряду происхождения
                filtered_lineage = list(compress(fulfilled_lineage, mask_lineage))
                for idx, f_rank in enumerate(filtered_lineage):
                    if f_rank['taxonId'] == target_id:
                        break
                    else:
                        touched_set.add((f_rank['taxonId'], f_rank['rank'], f_rank['scientificName'],
                                         filtered_lineage[idx + 1]['taxonId']))

    logger.debug(f"Total items in taxonomy: {len(touched_set)}")
    logger.debug(f"Have at least one peptide: {len(filtered_records_dict)}")
    logger.debug("Setting a DataFrame")

    df = pd.DataFrame(np.array(list(touched_set),
                               dtype=[("taxonId", "i4"), ("rank", 'O'), ("scientificName", 'O'), ("parentId", "i4")]))

    df.set_index("taxonId", drop=False, inplace=True)

    values_gen = df["taxonId"].parallel_apply(
        lambda x: ",".join(filtered_records_dict[x]) if filtered_records_dict.get(x) else None
    )

    df["peps"] = values_gen

    logger.debug("Creating model and fitting all the peptides")
    count_vect = CountVectorizer(tokenizer=lambda x: x.split(','), token_pattern=None)  # создаем модель
    count_vect.fit(at_least_all_peps)
    logger.debug("Remembering the peptides library for future extraction")
    library = count_vect.get_feature_names_out()

    max_level = target_info['rank']
    # основной цикл для расчетов
    for working_on, tax_level, cut_level in zip(config['ORDERED_RANKING'][:-1],
                                                config['ORDERED_RANKING'][1:],
                                                config['iterable_ordered_cutting']):
        if working_on == max_level:
            logger.debug(f"Reached highest local tree point, exit")
            break

        # выделение подраздела текущего уровня дерева из вообще всех
        sub_df = df[(df["rank"] == working_on) & ~(df["peps"].isna())]
        logger.debug(f"Gathered {sub_df.shape[0]} species with peptides for level {working_on}, passing to matrix")
        X = count_vect.transform(sub_df["peps"].values.tolist())

        with open(args.out_pattern + f"_{tax_level}.tsv", 'w') as wr:
            wr.write(f"taxonId\tscientificName\t"
                     f"C100\tC{int(config['common_like_cut']*100)}\tC{int(cut_level*100)}\t" +
                     f"U100\tU{int(config['common_like_cut']*100)}\n")
            # итерируемся по родителям
            for parent_group in tqdm(df[df["rank"] == tax_level]["taxonId"].unique(), desc=tax_level):
                # parent_group == taxonId родителя
                # пишем первые два столбца
                wr.write(f"{parent_group}\t{df.loc[parent_group, 'scientificName']}\t")
                # получаем маску всех детей этого родителя
                children_boolean_mask = sub_df["parentId"] == parent_group

                # получаем индексы ID детей вдоль пространства матрицы (ось 0)
                children_indexes = sub_df.index.get_indexer(sub_df[children_boolean_mask]["taxonId"])
                # получаем индексы всех НЕ детей вдоль пространства матрицы (ось 0)
                out_group_indexes = sub_df.index.get_indexer(sub_df[~children_boolean_mask]["taxonId"])

                # получаем вектор пептидов, представленных у всех детей данной группы
                strictly_common = X[children_indexes].min(axis=0).toarray()[0]
                # получаем ненулевые индексы общих пептидов
                non_zero_strict = np.nonzero(strictly_common)[0]
                # пишем пептиды, которые являются общими для всей группы
                wr.write(",".join(library[non_zero_strict]) + "\t")

                # вычисляем средний показатель вхождения пептидов в группу
                mean_values = np.array(X[children_indexes].mean(axis=0))[0]
                # индексы тех пептидов, что входят в 80% всех представителей
                common_like = np.where(mean_values > config['common_like_cut'])[0]
                # пишем пептиды, которые являются слабо, но общими для всей группы
                wr.write(",".join(library[common_like]) + "\t")

                # индексы тех пептидов, что встречаются более чем в cut_level случаев
                upper_than_border = np.where(mean_values > cut_level)[0]
                # пишем пептиды, которые присваиваем этому таксономическому уровню от потомков
                wr.write(",".join(library[upper_than_border]) + "\t")

                # если у нас на текущем уровне есть такие таксоны, что не являются близкими родственниками
                # (такое случается, когда мы подбираемся к самому верху нашего локального дерева)
                if len(out_group_indexes) > 0:
                    # получаем вектор пептидов, представленных у любых НЕ детей данной группы
                    out_group_peps = X[out_group_indexes].max(axis=0).toarray()[0]
                    # получаем ненулевые индесы всех внешних пептидов
                    non_zero_out_group_peps = np.nonzero(out_group_peps)[0]
                    # получаем индексы всех уникальных для группы детей пептидов
                    masked_unique = non_zero_strict[~np.isin(non_zero_strict, non_zero_out_group_peps)]
                    wr.write(",".join(library[masked_unique]) + "\t")

                    # индексы уникальных пептидов, входящих в 80% всех представителей
                    masked_unique_common_like = common_like[~np.isin(common_like, non_zero_out_group_peps)]
                    wr.write(",".join(library[masked_unique_common_like]) + "\t")
                # если у нас нет внешних таксонов, просто пишем 2 пустых колонки
                else:
                    wr.write("\t\t")
                # не забываем записать переход на новую строку
                wr.write("\n")
                # присваиваем родителю набор пептидов из дочерних
                df.loc[parent_group, "peps"] = ",".join(library[upper_than_border])

    logger.info("Done normally, end.")
