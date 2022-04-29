# ComputerVisionT2
Repositório para armazenamento do Trabalho de Implementação 2 da cadeira de Visão Computacional.

Este trabalho tem data final de entrega 29/04/22, a descrição das atividades que devem ser realizadas estão no arquivo [PDF](visao_trabalho2.pdf)

Relatório final [aqui](Relatorio_CMP_197_Trabalho_2.pdf).

## Executar

Para executar o script CBIR.py

```
cd scripts
python3 CBIR.py argv[1] argv[2] argv[3]
```

## ARGUMENTOSS PARA RODAR O PROGRAMA
    
- argv[1] = SEARCH_IMG (Somente o nome do arquivo de entrada sem a extensão)
- argv[2] = N (Quantidade de imagens a serem retornadas)
- argv[3] = SHOW_RESULT (0 - Não mostra a imagem com os correspondentes | 1 - Mostra a imagem com os correpondentes)

- Função CBIR: Faz a busca das N imagens mais semelheantes com a imagem de entrada;
- Função CBIR_MOD: Aplica uma escala e rotação, tanto na imagem de entrada quanto nas imagens de busca, faz a busca das N imagens mais semelheantes com a imagem de entrada;