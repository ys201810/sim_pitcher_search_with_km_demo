# coding=utf-8
import streamlit as st
from PIL import Image
import numpy as np
import pickle
import pathlib
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image


def set_texts(pitch_types):
    for pitch_type in pitch_types:
        input_num = st.text_input(pitch_type, '0')
        st.session_state[pitch_type] = input_num
    return True

def make_pt_scatter(pca_df, coordinate, near_cluster_num, image_path):
    fig = plt.figure(figsize=(8, 8))
    colors = ['blue', 'red', 'green', 'orange', 'violet']

    for i in range(5):  # [TODO] 5は外だし
        tmp = pca_df.loc[pca_df['cluster'] == i]
        mean_x = tmp[0].mean()
        mean_y = tmp[1].mean()
        plt.plot(mean_x, mean_y, marker='^', color=colors[i], markeredgecolor='black', markersize=10)
        plt.scatter(tmp[0], tmp[1], label=f'cluster{i}', color=colors[i])
    plt.legend()
    # 利用者の座標をプロット
    plt.plot(coordinate[0][0], coordinate[0][1], marker='*', color='black', markeredgecolor='black', markersize=16)
    pca_result_path = image_path / "pca_result.jpg"
    plt.savefig(pca_result_path)
    return pca_result_path

def search_similar_pitcher(pitch_type_nums, min_max_file_path,
                           kmeans_path, pca_df_path, pca_path, image_path, cluster_good_player_path):
    with open(min_max_file_path, 'rb') as inf:
        mm_scaler = pickle.load(inf)
    with open(kmeans_path, 'rb') as inf:
        kmeans = pickle.load(inf)
    with open(pca_df_path, 'rb') as inf:
        pca_df = pickle.load(inf)
    with open(pca_path, 'rb') as inf:
        pca = pickle.load(inf)
    with open(cluster_good_player_path, 'rb') as inf:
        cluster_good_player_df = pickle.load(inf)
    pitch_type_rates = [pt_num / sum(pitch_type_nums) for pt_num in pitch_type_nums]
    np_pitch_type_rates = np.array([pitch_type_rates])
    min_max_new_data = mm_scaler.transform(np_pitch_type_rates)

    cluster_distances = []
    for center_num, center in enumerate(kmeans.cluster_centers_):
        distance = np.linalg.norm(min_max_new_data - center)
        cluster_distances.append(distance)

    # 中心点からの距離が最も小さいクラスタ番号を取得(属するクラスタの特定)
    target_cluster_num = cluster_distances.index(min(cluster_distances))

    # PCAで描画
    target_coordinate = pca.transform(min_max_new_data)
    st.write(target_coordinate)

    pca_result_path = make_pt_scatter(pca_df, target_coordinate, target_cluster_num, image_path)
    pca_top3_path = pca_result_path.parent / "pca_top3_pt.png"

    st.write(f'あなたは、クラスタ{target_cluster_num}に属しています。')
    st.write(f'★があなたで、▲はクラスタの中心、●はメジャーリーグの各ピッチャーです。')
    st.image(Image.open(pca_result_path), use_column_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.header('クラスタ図')
        st.image(Image.open(pca_result_path), use_column_width=True)
    with col2:
        st.header('クラスタごとによく投げられる球種')
        st.image(Image.open(pca_top3_path), use_column_width=True)
    st.write(f'クラスタ{target_cluster_num}のWHIP(Walks and Hits per Inning Pitched)が優秀な選手は以下です。')
    st.table(cluster_good_player_df[target_cluster_num][['player_name', 'WHIP', 'W', 'L', 'ERA']])
    st.write(f'これらの選手の中から真似したい投手を選び、好調と不調の場合のピッチング傾向を把握して、好調時の癖を意識しながら練習してみましょう！')

def main():
    """
    球種の数を入力し、それを元に近い選出を判別する。
    :return:
    """
    base_path = pathlib.Path.cwd().resolve().parent

    st.title('試合で投げる球種数からメジャーリーグの誰に似ているか判別するよ！')

    st.markdown('#### 1. 右投げか左投げを選択(今は機能してないよ)')
    option = st.selectbox(
        '利き腕の選択',
        ['右投げ', '左投げ']
    )

    st.markdown('#### 2. 球種別の投球数を入力')
    pitch_types = ['スライダー', 'シンカー', 'チェンジアップ', 'カーブ', '4シーム', 'スプリットフィンガー',
                   'カッター', 'ナックルカーブ', 'スローカーブ', 'イーファス']
    set_texts(pitch_types)

    st.markdown('#### 3. 近い投手を検索！を押して開始')
    # 球種ごとの割合を計算
    button_state = st.button('近い投手を検索！')
    if button_state:
        try:
            pitch_type_nums = [int(st.session_state[val]) for val in pitch_types]
        except:
            st.error('全ての球種に数値を入力してください。')
        pitch_type_nums.insert(-2, 0)  # 'FA'という少量の謎pitch typeがあるので、これを0固定で入力
        min_max_file_path = base_path / "data" / "mm.pickle"
        kmeans_path = base_path / "data" / "km_model.pickle"
        pca_df_path = base_path / "data" / "pca_df.pickle"
        pca_path = base_path / "data" / "pca.pickle"
        cluster_good_player_path = base_path / "data" / "row_whip_dfs.pickle"
        image_path = base_path / "data" / "images"

        search_similar_pitcher(pitch_type_nums, min_max_file_path, kmeans_path, pca_df_path, pca_path,
                               image_path, cluster_good_player_path)
    else:
        st.write('Goodbye')

if __name__ == '__main__':
    main()
