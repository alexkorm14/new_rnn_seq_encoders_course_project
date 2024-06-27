# new_rnn_seq_encoders_course_project
 Usage of new rnn model in pytorch lifestream

 Краткая навигация:
 ◦ seq_encoders - папка с моделями (в файле модели приложена ссылка на гит источник) + new_rnn_encoder.py (класс RnnEncoder с добавлением новых архитектур) + simple_seq_encoder.py (для создания seq encoder на основании trx encoder + rnn encoder)
 
 ◦ Data - папка с данными исходными и предобработанными для моделирования
 
 ◦ make_data_config/make_model_config.ipynb - определить конфиги для удобства сохранения результатов и запуска
 
 ◦ Preprocessing_with_eda.ipynb - по заранее созданному конфигу датасета запустить предобработку и получить статистику по кол-во событий у id
 
 ◦ Coles_model_rnn.ipynb - запуск эксперимента по конфигу модели
 
 ◦ Coles_collect_results.ipynb - сбор результатов экспериментов
 
 ◦ Остальные файлы напоминают Coles_model_rnn.ipynb только для задачи sequence to target или чтобы запустить трансформер


Данные хранятся на гугл диске: https://drive.google.com/drive/folders/1VA46oZdIjZxTXH1bnY_tr60b0ywWzfSv?usp=drive_link
