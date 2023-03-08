# P_IA368DD_2023S1
Repositório com os exercícios da Unicamp

# Exercício 1

Enunciado do exercício da semana:

1. Usar o BM25 implementado pelo pyserini para buscar queries no TREC-DL 2020 ([Documentação de referência](https://github.com/castorini/pyserini/blob/master/docs/experiments-msmarco-passage.md))
2. Implementar um buscador booleano/bag-of-words.
3. Implementar um buscador com TF-IDF
4. Avaliar implementações 1, 2, e 3 no TREC-DL 2020 e calcular o nDCG@10

Nos itens 2 e 3: (i) Fazer uma implementação que suporta buscar eficientemente milhões de documentos e (ii) Não se pode usar bibliotecas como sklearn, que já implementam o BoW e TF-IDF.