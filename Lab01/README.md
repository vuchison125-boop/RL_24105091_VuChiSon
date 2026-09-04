# Bài thực hành số 1 – Làm quen với Gymnasium

## Thông tin sinh viên

- Họ tên: Vu Chi Son
- MSSV:24105091
- Lớp: EEE-AI K18
- GitHub username:  vuchison125-boop
- Repository URL: RL_24105091_VuChiSon

## Môi trường thực thi

- Python version: 3.13.2
- Gymnasium version: 1.3.0
- NumPy version: 2.5.2
- Matplotlib version: 3.11.1

## Mục tiêu

Bài thực hành này nhằm làm quen với môi trường học tăng cường và thư viện Gymnasium.

Các nội dung chính:

- Tạo và sử dụng môi trường RL.
- Làm việc với `action_space` và `observation_space`.
- Sử dụng API `reset()` và `step()` của Gymnasium.
- Phân biệt `terminated` và `truncated`.
- Sử dụng random seed để tái lập thí nghiệm.
- Xây dựng random agent.
- Thu thập và phân tích reward qua nhiều episode.
- Xây dựng và đánh giá các policy đơn giản.
- Làm quen với môi trường FrozenLake.
- Tổ chức và quản lý mã nguồn bằng Git và GitHub.

## Cấu trúc thư mục

```text
Lab01/
├── README.md
├── requirements.txt
├── src/
│   ├── bai01.py
│   ├── bai02.py
│   ├── ...
│   ├── bai36.py
│   ├── main.py
│   └── migration_gym_to_gymnasium.py
├── notebooks/
│   └── Lab01_MSSV_HoTen.ipynb
├── figures/
│   ├── reward_cartpole.png
│   ├── moving_average.png
│   └── comparison_agents.png
└── data/
    └── README.md