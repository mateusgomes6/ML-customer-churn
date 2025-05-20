# ML-customer-churn


/ML-CHURN-DE-CLIENTES
│   README.md
│   main.py                     # Ponto de entrada principal
|   dataframe.py                # DataFrame usado para estudo do churn
│   .gitignore
|   
├───src
│   │   __init__.py             # Tornar o diretório um pacote Python
│   │
│   ├───data
│   │       preprocessing.py    # Funções de pré-processamento
│   │       visualization.py    # Funções de visualização
│   │
│   ├───models
│   │       clustering.py       # Funções de clusterização
│   │       classification.py   # Modelo de classificação
│   │       predict.py         # Funções de predição
│   │
│   └───utils
│           helpers.py          # Funções auxiliares
│
└───tests
    │   conftest.py            # Configurações de teste
    │
    ├───unit
    │       test_preprocessin.py
    │       test_helpers.py
    │
    └───integration
            test_pipeline.py
