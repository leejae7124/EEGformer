from argparse import ArgumentParser
from utils import log2csv, ensure_path
from Task import LOSO
import os


parser = ArgumentParser()
parser.add_argument('--full-run', type=int, default=1, help='If it is set as 1, you will run LOSO on the same machine.') #전체 피험자 대상으로 실행
parser.add_argument('--test-sub', type=int, default=0, help='If full-run is set as 0, you can use this to leave this ' # full-run = 0 일 때, 테스트할 피험자 인덱스
                                                            'subject only. Then you can divided LOSO on different'
                                                            ' machines')
######## Data ########
parser.add_argument('--dataset', type=str, default='FATIG')
parser.add_argument('--subjects', type=int, default=11)
parser.add_argument('--num-class', type=int, default=2, choices=[2, 3, 4])
parser.add_argument('--label-type', type=str, default='FTG')
parser.add_argument('--num-chan', type=int, default=30) # 24 for TSception
parser.add_argument('--num-time', type=int, default=384)
parser.add_argument('--segment', type=int, default=4, help='segment length in seconds')
parser.add_argument('--trial-duration', type=int, default=60, help='trial duration in seconds')
parser.add_argument('--overlap', type=float, default=0)
parser.add_argument('--sampling-rate', type=int, default=128)
parser.add_argument('--data-format', type=str, default='eeg')
######## Training Process ########
parser.add_argument('--random-seed', type=int, default=2023)
parser.add_argument('--max-epoch', type=int, default=200)
parser.add_argument('--additional-epoch', type=int, default=20)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--val-rate', type=float, default=0.2)

parser.add_argument('--save-path', default='./save_att2/') # change this
parser.add_argument('--load-path', default='./data_processed/') # change this
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--mixed-precision', type=int, default=0)
######## Model Parameters ########
parser.add_argument('--model', type=str, default='Deformer')
parser.add_argument('--graph-type', type=str, default='BL', choices=['LGG-G', 'LGG-F', 'LGG-H', 'TS', 'BL'])
parser.add_argument('--kernel-length', type=int, default=13)
parser.add_argument('--T', type=int, default=64)
parser.add_argument('--AT', type=int, default=16)
parser.add_argument('--num-layers', type=int, default=6)


args = parser.parse_args()
all_sub_list = [0, 4, 21, 30, 34, 40, 41, 42, 43, 44, 52] #피험자 리스트 정의

if args.model == 'TSception': #모델에 따른 하이퍼파라미터 유효성 검사, TSception 모델일 경우, 입력 형식이 맞는지 확인
    assert args.graph_type == 'TS', "When using TSception, suppose to get graph_type of 'TS'," \
                                    " but get {} instead!".format(args.graph_type)
    assert args.num_chan == 24, "When using TSception, suppose to have num_chan==24," \
                                " but get {} instead!".format(args.num_chan)

if args.model == 'LGGNet':
    assert args.graph_type in ['LGG-G', 'LGG-F', 'LGG-H'], "When using LGGNet, suppose to get graph_type " \
                                                           "of 'LGG-X'(X=G, F, or H), but get {} " \
                                                           "instead!".format(args.graph_type)

if args.full_run: #실행할 피험자 리스트 설정
    sub_to_run = all_sub_list
else:
    sub_to_run = [args.test_sub]

logs_name = 'logs_{}_{}'.format(args.dataset, args.model)
for sub in sub_to_run: #피험자별 실험 실행 루프
    results = LOSO( #LOSO 방식으로 하나의 피험자를 테스트로 두고, 나머지는 학습 데이터로 사용. LOSO: 피험자 단위의 교차 검증 방법.
        test_idx=[sub], subjects=all_sub_list,#반복하면 각 피험자에 대해 학습-검증-테스트 수행
        experiment_ID='sub{}'.format(sub), args=args, logs_name=logs_name
    )
    log_path = os.path.join(args.save_path, logs_name, 'sub{}'.format(sub))
    ensure_path(log_path)
    log2csv(os.path.join(log_path, 'result.csv'), results[0]) #실험 결과를 csv 형식으로 저장