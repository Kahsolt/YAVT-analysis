#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/06/08 

import json
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt
#plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from sklearn.decomposition import PCA

BASE_PATH = Path(__file__).absolute().parent
REPO_PATH = BASE_PATH / 'repo' / 'yavt.fengtao.xyz'
QUEST_PATH = REPO_PATH / 'questions'
MODEL_PATH = REPO_PATH / 'models'

QUEST_FILES = {
  '8values': QUEST_PATH / '8-values.json',
  'yavt010': QUEST_PATH / 'yavt_0_1_0.json',
  'yavt021': QUEST_PATH / 'yavt_0_2_1.json',
}
MODEL_FILES = {
  '8values': MODEL_PATH / '8-values.json',
  'yavt':    MODEL_PATH / 'yavt-5-axis.json',
}


def go(args):
  ''' Model '''
  with open(MODEL_FILES[args.model], 'r', encoding='utf-8') as fh:
    model = json.load(fh)

  ideologies = model['ideologies']
  X = np.stack(ideologies.values(), axis=0)
  Y = list(ideologies.keys())
  print('ideologies X.shape:', X.shape)

  if not 'pca 2-dim':
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    print('explained_variance:', pca.explained_variance_)
    print('explained_variance_ratio:', pca.explained_variance_ratio_, f'{sum(pca.explained_variance_ratio_):.3f}')

    plt.clf()
    plt.scatter(X_pca[:, 0], X_pca[:, 1])
    for i, (x, y) in enumerate(zip(X_pca[:, 0], X_pca[:, 1])):
      plt.text(x-args.offset, y-args.offset, Y[i])
    plt.suptitle('ideologies 2-dim')
    plt.tight_layout()
    plt.show()

    plt.clf()
    ax = plt.axes(projection='3d')
    ax.scatter3D(X_pca[:, 0], X_pca[:, 1], np.zeros_like(X_pca[:, 0]), label=Y)
    for i, (x, y, z) in enumerate(zip(X_pca[:, 0], X_pca[:, 1], np.zeros_like(X_pca[:, 0]))):
      ax.text(x-args.offset, y-args.offset, z, Y[i])
    plt.suptitle('ideologies 3-dim')
    plt.tight_layout()
    plt.axis('off')
    plt.show()

  if 'pca 3-dim':
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)
    print('explained_variance:', pca.explained_variance_)
    print('explained_variance_ratio:', pca.explained_variance_ratio_, f'{sum(pca.explained_variance_ratio_):.3f}')

    plt.clf()
    ax = plt.axes(projection='3d')
    ax.scatter3D(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], label=Y)
    for i, (x, y, z) in enumerate(zip(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2])):
      ax.text(x-args.offset, y-args.offset, z, Y[i])
    plt.suptitle('ideologies 3-dim')
    plt.tight_layout()
    plt.show()


  ''' Quest '''
  with open(QUEST_FILES[args.quest], 'r', encoding='utf-8') as fh:
    quest = json.load(fh)

  questions = quest['questions']
  X = np.stack([q['evaluation'] for q in questions], axis=0)
  print('questions X.shape:', X.shape)

  if 'pca 2-dim':
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    print('explained_variance:', pca.explained_variance_)
    print('explained_variance_ratio:', pca.explained_variance_ratio_, f'{sum(pca.explained_variance_ratio_):.3f}')

    plt.clf()
    plt.scatter(X_pca[:, 0], X_pca[:, 1])
    plt.suptitle('questions 2-dim')
    plt.tight_layout()
    plt.show()

  if 'pca 3-dim':
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)
    print('explained_variance:', pca.explained_variance_)
    print('explained_variance_ratio:', pca.explained_variance_ratio_, f'{sum(pca.explained_variance_ratio_):.3f}')

    plt.clf()
    ax = plt.axes(projection='3d')
    ax.scatter3D(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], label=Y)
    plt.suptitle('questions 3-dim')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-Q', '--quest', default='8values', choices=['8values', 'yavt010', 'yavt021'])
  parser.add_argument('--offset', default=5, type=int)
  args = parser.parse_args()

  args.model = 'yavt' if args.quest.startswith('yavt') else '8values'

  go(args)
