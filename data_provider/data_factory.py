from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
}
'''
Импортируются несколько классов датасетов (Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred) из модуля data_loader в проекте.
Создается словарь data_dict, который сопоставляет идентификаторы датасетов ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'custom') с соответствующими классами датасетов.
Определяется функция data_provider, которая возвращает датасет и загрузчик данных на основе переданных аргументов и флага ('train', 'test', 'pred').
Внутри функции data_provider определяются параметры для загрузчика данных в зависимости от переданного флага (flag). Например, для тестирования (flag='test') устанавливаются параметры для тестового набора данных, для предсказания (flag='pred') - параметры для предсказательного набора данных.
Создается экземпляр датасета (data_set) соответствующего класса на основе переданных параметров.
Создается загрузчик данных (data_loader) на основе созданного датасета с заданными параметрами.
Функция возвращает экземпляр датасета и загрузчик данных.'''

def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
