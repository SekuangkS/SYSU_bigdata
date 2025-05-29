"""
简历推荐系统推理引擎
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import json
import jieba
import os
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


def tokenize_chinese(text):
    """中文分词"""
    if not text or text == 'NULL' or pd.isna(text):
        return []
    return jieba.lcut(str(text))


class DeepRecommendationModel(nn.Module):
    """深度推荐模型"""

    def __init__(self, resume_feature_dim, job_feature_dim, hidden_dims=[512, 256, 128], use_sigmoid=True):
        super(DeepRecommendationModel, self).__init__()

        self.resume_feature_dim = resume_feature_dim
        self.job_feature_dim = job_feature_dim
        self.use_sigmoid = use_sigmoid

        # 简历编码器
        self.resume_encoder = nn.Sequential(
            nn.Linear(resume_feature_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # 职位编码器
        self.job_encoder = nn.Sequential(
            nn.Linear(job_feature_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # 交互层
        interaction_layers = [
            nn.Linear(hidden_dims[1] * 2, hidden_dims[2]),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[2], 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ]

        if self.use_sigmoid:
            interaction_layers.append(nn.Sigmoid())

        self.interaction_layer = nn.Sequential(*interaction_layers)

    def forward(self, resume_features, job_features):
        resume_encoded = self.resume_encoder(resume_features)
        job_encoded = self.job_encoder(job_features)
        combined = torch.cat([resume_encoded, job_encoded], dim=1)
        output = self.interaction_layer(combined)
        return output


class ResumeRecommendationInference:
    """简历推荐推理引擎"""

    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.skill_vectorizer = None
        self.scaler = None
        self.resume_data = None
        self.resume_features_cache = None

    def load_model_and_preprocessors(self):
        """加载模型和预处理器"""
        try:
            model_path = f'{self.model_dir}/recommendation_model.pth'
            preprocessor_path = f'{self.model_dir}/preprocessors.pkl'
            preprocessor_params_path = f'{self.model_dir}/preprocessor_params.json'

            if not os.path.exists(model_path):
                return False

            # 加载模型
            model_checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            if 'model_config' not in model_checkpoint:
                return False

            model_config = model_checkpoint['model_config']
            self.model = DeepRecommendationModel(
                resume_feature_dim=model_config['resume_feature_dim'],
                job_feature_dim=model_config['job_feature_dim'],
                hidden_dims=model_config.get('hidden_dims', [512, 256, 128]),
                use_sigmoid=model_config.get('use_sigmoid', True)
            )

            self.model.load_state_dict(model_checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()

            # 加载预处理器 - 优先JSON重建
            preprocessors_loaded = False

            if os.path.exists(preprocessor_params_path):
                try:
                    with open(preprocessor_params_path, 'r', encoding='utf-8') as f:
                        params = json.load(f)

                    self.skill_vectorizer = TfidfVectorizer(
                        max_features=params['vectorizer_params'].get('max_features', 1000),
                        tokenizer=tokenize_chinese
                    )

                    if params['vectorizer_params']['vocabulary_']:
                        self.skill_vectorizer.vocabulary_ = params['vectorizer_params']['vocabulary_']
                        self.skill_vectorizer.idf_ = np.array(params['vectorizer_params']['idf_'])
                        self.skill_vectorizer._tfidf = True

                    self.scaler = StandardScaler()
                    if params['scaler_params']['mean_']:
                        self.scaler.mean_ = np.array(params['scaler_params']['mean_'])
                        self.scaler.scale_ = np.array(params['scaler_params']['scale_'])
                        self.scaler.var_ = np.array(params['scaler_params']['var_'])
                        self.scaler.n_features_in_ = params['scaler_params']['n_features_in_']
                        self.scaler.n_samples_seen_ = params['scaler_params']['n_samples_seen_']

                    preprocessors_loaded = True
                except Exception:
                    pass

            # 备选方案：直接加载pickle
            if not preprocessors_loaded and os.path.exists(preprocessor_path):
                try:
                    with open(preprocessor_path, 'rb') as f:
                        preprocessors = pickle.load(f)
                    self.skill_vectorizer = preprocessors.get('skill_vectorizer')
                    self.scaler = preprocessors.get('scaler')
                    if self.skill_vectorizer is not None and self.scaler is not None:
                        preprocessors_loaded = True
                except Exception:
                    pass

            # 默认预处理器
            if not preprocessors_loaded:
                self.skill_vectorizer = TfidfVectorizer(max_features=1000, tokenizer=tokenize_chinese)
                self.scaler = StandardScaler()

            return True

        except Exception:
            return False

    def load_resume_data(self, data_path='data/Chinese_resume_data.csv'):
        """加载简历数据"""
        try:
            try:
                self.resume_data = pd.read_csv(data_path, encoding='utf-8')
            except UnicodeDecodeError:
                self.resume_data = pd.read_csv(data_path, encoding='gbk')

            self.resume_data = self.resume_data.fillna('NULL')

            # 构建技能特征
            skill_columns = ['编程语言', '前端技术', '后端技术', '数据库', '云计算/运维', '数据与算法', '移动开发', '测试工具']
            existing_skill_columns = [col for col in skill_columns if col in self.resume_data.columns]

            if not existing_skill_columns:
                self.resume_data['综合技能'] = 'Python Java'
            else:
                self.resume_data['综合技能'] = self.resume_data[existing_skill_columns].apply(
                    lambda row: ' '.join([str(val) for val in row if val != 'NULL']), axis=1
                )

            # 计算评分
            self.resume_data['工作经验评分'] = self._calculate_experience_score()
            self.resume_data['项目经验评分'] = self._calculate_project_score()

            education_mapping = {'专科': 1, '本科': 2, '硕士及以上': 3}
            self.resume_data['学历评分'] = self.resume_data['学历层次'].map(education_mapping).fillna(1)

            self._precompute_resume_features()
            return True

        except Exception:
            return False

    def _calculate_experience_score(self):
        """计算工作经验评分"""
        def score_experience(row):
            score = 0
            exp_mapping = {'1年以下': 0.5, '1―3年': 2, '3―5年': 4, '5年以上': 6}
            exp_columns = ['小型企业工作经验', '中型企业工作经验', '大型企业工作经验']
            existing_exp_columns = [col for col in exp_columns if col in row.index]

            if not existing_exp_columns:
                return 2.0

            for exp_type in existing_exp_columns:
                if row[exp_type] != 'NULL':
                    exp_value = str(row[exp_type])
                    if exp_type == '大型企业工作经验':
                        score += exp_mapping.get(exp_value, 0) * 1.5
                    elif exp_type == '中型企业工作经验':
                        score += exp_mapping.get(exp_value, 0) * 1.2
                    else:
                        score += exp_mapping.get(exp_value, 0)
            return max(score, 1.0)

        return self.resume_data.apply(score_experience, axis=1)

    def _calculate_project_score(self):
        """计算项目经验评分"""
        project_columns = ['小规模项目', '中规模项目', '大规模项目']
        existing_project_columns = [col for col in project_columns if col in self.resume_data.columns]

        if len(existing_project_columns) == 3:
            return (pd.to_numeric(self.resume_data['小规模项目'], errors='coerce').fillna(0) * 1 +
                    pd.to_numeric(self.resume_data['中规模项目'], errors='coerce').fillna(0) * 2 +
                    pd.to_numeric(self.resume_data['大规模项目'], errors='coerce').fillna(0) * 4).clip(lower=1.0)
        else:
            return pd.Series([2.0] * len(self.resume_data), index=self.resume_data.index)

    def _precompute_resume_features(self):
        """预计算简历特征向量"""
        try:
            if not hasattr(self.skill_vectorizer, 'vocabulary_') or not self.skill_vectorizer.vocabulary_:
                skill_vectors = self.skill_vectorizer.fit_transform(self.resume_data['综合技能']).toarray()
            else:
                skill_vectors = self.skill_vectorizer.transform(self.resume_data['综合技能']).toarray()

            resume_features = []
            for idx, row in self.resume_data.iterrows():
                features = list(skill_vectors[idx]) + [
                    row['学历评分'],
                    row['工作经验评分'],
                    row['项目经验评分'],
                    float(row['年龄']) if str(row['年龄']).replace('.', '').isdigit() else 25.0,
                    1.0 if row['性别'] == '男' else 0.0
                ]
                resume_features.append(features)

            if not hasattr(self.scaler, 'mean_') or self.scaler.mean_ is None:
                self.resume_features_cache = self.scaler.fit_transform(np.array(resume_features))
            else:
                self.resume_features_cache = self.scaler.transform(np.array(resume_features))

        except Exception as e:
            raise e

    def _build_job_features(self, job_requirements: Dict) -> np.ndarray:
        """构建职位特征向量"""
        required_skills = job_requirements.get('required_skills', '')
        skill_vector = self.skill_vectorizer.transform([required_skills]).toarray()[0]

        min_education = job_requirements.get('min_education', '本科')
        education_mapping = {'专科': 1, '本科': 2, '硕士及以上': 3}
        education_score = education_mapping.get(min_education, 2)

        min_experience = job_requirements.get('min_experience_years', 2)
        max_age = job_requirements.get('max_age', 35)

        job_features = list(skill_vector) + [education_score, min_experience, max_age]
        return np.array(job_features)

    def recommend(self, job_requirements: Dict, top_k: int = 10) -> List[Dict]:
        """推荐匹配简历"""
        if self.model is None or self.resume_features_cache is None:
            raise ValueError("模型或数据未加载")

        try:
            job_features = self._build_job_features(job_requirements)
            filtered_indices = self._filter_basic_requirements(job_requirements)

            if len(filtered_indices) == 0:
                return []

            filtered_resume_features = self.resume_features_cache[filtered_indices]
            job_features_batch = np.tile(job_features, (len(filtered_indices), 1))

            # 模型推理
            with torch.no_grad():
                resume_tensor = torch.FloatTensor(filtered_resume_features).to(self.device)
                job_tensor = torch.FloatTensor(job_features_batch).to(self.device)
                scores = self.model(resume_tensor, job_tensor).cpu().numpy().flatten()

            # 构建推荐结果
            recommendations = []
            for i, idx in enumerate(filtered_indices):
                resume = self.resume_data.iloc[idx]
                recommendations.append({
                    'resume_id': int(resume['简历编号']),
                    'name': str(resume['姓名']),
                    'gender': str(resume['性别']),
                    'age': int(resume['年龄']) if str(resume['年龄']).isdigit() else 0,
                    'position': str(resume['意向岗位']),
                    'education': str(resume['学历层次']),
                    'skills': str(resume['综合技能']),
                    'screening_result': str(resume.get('筛选结果', '')),
                    'score': float(scores[i])
                })

            recommendations.sort(key=lambda x: x['score'], reverse=True)
            return recommendations[:top_k]

        except Exception:
            return []

    def _filter_basic_requirements(self, job_requirements: Dict) -> List[int]:
        """根据基本要求筛选简历"""
        filtered_data = self.resume_data.copy()

        try:
            if job_requirements.get('position'):
                filtered_data = filtered_data[filtered_data['意向岗位'] == job_requirements['position']]

            if job_requirements.get('min_education'):
                education_order = ['专科', '本科', '硕士及以上']
                min_education = job_requirements['min_education']
                if min_education in education_order:
                    min_edu_idx = education_order.index(min_education)
                    valid_educations = education_order[min_edu_idx:]
                    filtered_data = filtered_data[filtered_data['学历层次'].isin(valid_educations)]

            if job_requirements.get('max_age'):
                max_age = int(job_requirements['max_age'])
                filtered_data = filtered_data[pd.to_numeric(filtered_data['年龄'], errors='coerce') <= max_age]

            if job_requirements.get('min_age'):
                min_age = int(job_requirements['min_age'])
                filtered_data = filtered_data[pd.to_numeric(filtered_data['年龄'], errors='coerce') >= min_age]

            return filtered_data.index.tolist()

        except Exception:
            return list(range(len(self.resume_data)))

    def search_resumes(self, search_params: Dict) -> List[Dict]:
        """搜索简历"""
        try:
            filtered_data = self.resume_data.copy()

            if search_params.get('position'):
                filtered_data = filtered_data[filtered_data['意向岗位'] == search_params['position']]

            if search_params.get('education'):
                filtered_data = filtered_data[filtered_data['学历层次'] == search_params['education']]

            if search_params.get('gender'):
                filtered_data = filtered_data[filtered_data['性别'] == search_params['gender']]

            if search_params.get('min_age'):
                filtered_data = filtered_data[pd.to_numeric(filtered_data['年龄'], errors='coerce') >= search_params['min_age']]

            if search_params.get('max_age'):
                filtered_data = filtered_data[pd.to_numeric(filtered_data['年龄'], errors='coerce') <= search_params['max_age']]

            if search_params.get('skills'):
                skill_keywords = search_params['skills'].lower()
                mask = filtered_data['综合技能'].str.lower().str.contains(skill_keywords, na=False)
                filtered_data = filtered_data[mask]

            limit = min(search_params.get('limit', 20), 100)
            results = filtered_data.head(limit)

            resume_list = []
            for _, resume in results.iterrows():
                resume_list.append({
                    'resume_id': int(resume['简历编号']),
                    'name': str(resume['姓名']),
                    'gender': str(resume['性别']),
                    'age': int(resume['年龄']) if str(resume['年龄']).isdigit() else 0,
                    'position': str(resume['意向岗位']),
                    'education': str(resume['学历层次']),
                    'skills': str(resume['综合技能']),
                    'screening_result': str(resume.get('筛选结果', ''))
                })

            return resume_list

        except Exception:
            return []

    def get_resume_details(self, resume_id: int) -> Dict:
        """获取简历详细信息"""
        try:
            resume = self.resume_data[self.resume_data['简历编号'] == resume_id]
            if resume.empty:
                return {}

            resume = resume.iloc[0]

            def safe_get(field, default=''):
                value = resume.get(field, default)
                if pd.isna(value) or value == 'NULL':
                    return default
                return str(value)

            def safe_get_int(field, default=0):
                try:
                    value = resume.get(field, default)
                    if pd.isna(value) or value == 'NULL':
                        return default
                    return int(value)
                except (ValueError, TypeError):
                    return default

            return {
                'resume_id': int(resume['简历编号']),
                'name': safe_get('姓名'),
                'gender': safe_get('性别'),
                'age': safe_get_int('年龄'),
                'position': safe_get('意向岗位'),
                'education': safe_get('学历层次'),
                'skills': safe_get('综合技能'),
                'screening_result': safe_get('筛选结果')
            }

        except Exception:
            return {}

    def get_statistics(self) -> Dict:
        """获取数据统计信息"""
        if self.resume_data is None:
            return {}

        try:
            stats = {
                'total_resumes': len(self.resume_data),
                'positions': self.resume_data['意向岗位'].value_counts().to_dict(),
                'education_levels': self.resume_data['学历层次'].value_counts().to_dict(),
                'age_stats': {
                    'mean': float(pd.to_numeric(self.resume_data['年龄'], errors='coerce').mean()),
                    'min': int(pd.to_numeric(self.resume_data['年龄'], errors='coerce').min()),
                    'max': int(pd.to_numeric(self.resume_data['年龄'], errors='coerce').max())
                }
            }
            return stats

        except Exception:
            return {}

    def initialize(self, data_path='data/Chinese_resume_data.csv'):
        """初始化推理引擎"""
        if not self.load_model_and_preprocessors():
            return False
        if not self.load_resume_data(data_path):
            return False
        return True


# 全局推理引擎实例
inference_engine = None

def get_inference_engine():
    """获取推理引擎单例"""
    global inference_engine
    if inference_engine is None:
        inference_engine = ResumeRecommendationInference()
        if not inference_engine.initialize():
            raise RuntimeError("推理引擎初始化失败")
    return inference_engine


if __name__ == "__main__":
    engine = ResumeRecommendationInference()
    if engine.initialize():
        job_requirements = {
            'position': '后端开发工程师',
            'required_skills': 'Java Python Spring MySQL',
            'min_education': '本科',
            'min_experience_years': 2,
            'max_age': 35
        }
        recommendations = engine.recommend(job_requirements, top_k=5)
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['name']} - 评分: {rec['score']:.4f}")