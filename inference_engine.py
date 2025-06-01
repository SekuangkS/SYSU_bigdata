"""
简历推荐系统推理引擎
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import json
import jieba
import os
import joblib
from ruamel.yaml import YAML
from typing import List, Dict
from src.preprocessors import ResumePreprocessor, tokenize_chinese, TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import warnings
import traceback

warnings.filterwarnings('ignore')


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


class CoarseRanker:
    """粗排模块：使用轻量级模型快速筛选候选集"""

    def __init__(self, model_path='models/coarse_ranker.pkl'):
        self.model = None
        self.load_model(model_path)

    def load_model(self, model_path):
        """加载预训练的粗排模型"""
        try:
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
            else:
                # 创建默认模型并训练简单版本
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.datasets import make_classification
                X, y = make_classification(n_samples=100, n_features=5, random_state=42)
                self.model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
                self.model.fit(X, y)
                joblib.dump(self.model, model_path)
                print(f"创建并保存了默认粗排模型到 {model_path}")
        except Exception as e:
            print(f"加载粗排模型失败: {e}")
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.datasets import make_classification
            X, y = make_classification(n_samples=100, n_features=5, random_state=42)
            self.model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
            self.model.fit(X, y)

    def extract_features(self, resume, job_req):
        """提取粗排特征"""
        try:
            # 加载预处理器参数获取特征维度
            preprocessor_path = os.path.join('models', 'preprocessor_params.json')
            if os.path.exists(preprocessor_path):
                with open(preprocessor_path, 'r', encoding='utf-8') as f:
                    params = json.load(f)
                feature_dim = params.get('feature_dim', 128)  # 默认128维
            else:
                feature_dim = 128  # 默认值
            
            # 确保所有特征都存在
            resume_skills = resume.get('skills', '')
            if isinstance(resume_skills, str):
                resume_skills = resume_skills.split()

            job_skills = job_req.get('required_skills', '')
            if isinstance(job_skills, str):
                job_skills = job_skills.split()

            # 计算技能匹配度
            common_skills = set(resume_skills) & set(job_skills)
            skill_match = len(common_skills) / max(len(job_skills), 1) if job_skills else 0.0

            # 学历匹配
            education_mapping = {'专科': 1, '本科': 2, '硕士及以上': 3}
            resume_edu = education_mapping.get(resume.get('education', '本科'), 2)
            job_min_edu = education_mapping.get(job_req.get('min_education', '本科'), 2)
            edu_match = 1.0 if resume_edu >= job_min_edu else 0.0

            # 经验匹配
            resume_exp = resume.get('experience', 0)
            job_min_exp = job_req.get('min_experience_years', 0)
            exp_match = min(resume_exp / max(job_min_exp, 1), 1.5)  # 经验匹配度，最高1.5倍

            # 年龄匹配
            resume_age = resume.get('age', 30)
            job_max_age = job_req.get('max_age', 35)
            age_match = min(resume_age / job_max_age, 1.0)  # 年龄越小越匹配

            # 职位匹配
            position_match = 1.0 if resume.get('position', '') == job_req.get('position', '') else 0.0

            # 构建128维特征向量
            features = np.zeros(128)
            
            # 填充已知特征
            features[0] = position_match
            features[1] = edu_match
            features[2] = exp_match
            features[3] = age_match
            features[4] = skill_match
            
            return features.reshape(1, -1)
            
        except Exception as e:
            print(f"提取粗排特征失败: {e}")
            # 返回默认特征向量
            return np.zeros((1, 128))

    def predict(self, resume, job_req):
        """预测粗排分数"""
        try:
            features = self.extract_features(resume, job_req)
            if hasattr(self.model, "predict_proba"):
                return self.model.predict_proba(features)[0][1]
            else:
                # 如果没有概率预测，使用决策函数或简单预测
                return self.model.predict(features)[0]
        except Exception as e:
            print(f"粗排预测错误: {e}")
            return 0.5  # 默认值


class ReRanker:
    """重排模块：对精排结果进行业务规则调整"""

    def __init__(self, config_path='config.yaml'):
        self.config = self.load_config(config_path)
        self.diversity_rules = self.config.get('reranking', {}).get('diversity', {})
        self.business_rules = self.config.get('reranking', {}).get('business_rules', {})

    def load_config(self, config_path):
        """加载YAML配置文件"""
        try:
            yaml = YAML(typ='safe')  # 使用安全加载器
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    return yaml.load(f)
        except Exception as e:
            print(f"加载配置文件失败: {e}")

        # 默认配置
        return {
            'reranking': {
                'diversity': {
                    'max_per_position': 3,
                    'min_results': 5,
                    'skill_coverage': True
                },
                'business_rules': {
                    'blacklist': [],
                    'priority_conditions': [
                        {
                            'education': "硕士及以上",
                            'experience': 5
                        }
                    ]
                }
            }
        }
    def apply_diversity(self, recommendations):
        """应用多样性规则"""
        if not recommendations:
            return []

        # 1. 岗位多样性
        position_count = {}
        reranked = []

        for rec in recommendations:
            position = rec.get('position', '')
            if position not in position_count:
                position_count[position] = 0

            if position_count[position] < self.diversity_rules.get('max_per_position', 3):
                reranked.append(rec)
                position_count[position] += 1

        # 2. 技能多样性 (可选)
        if self.diversity_rules.get('skill_coverage', True):
            skill_coverage = set()
            final_list = []

            for rec in reranked:
                skills = rec.get('skills', '')
                if isinstance(skills, str):
                    skills = skills.split()

                new_skills = set(skills) - skill_coverage
                if new_skills or len(final_list) < self.diversity_rules.get('min_results', 5):
                    final_list.append(rec)
                    skill_coverage.update(new_skills)
            return final_list

        return reranked

    def apply_business_rules(self, recommendations):
        """应用业务规则"""
        if not recommendations:
            return []

        reranked = list(recommendations)

        # 1. 特殊人才置顶
        for condition in self.business_rules.get('priority_conditions', []):
            for i, rec in enumerate(reranked):
                match = True

                # 检查学历条件
                if 'education' in condition:
                    rec_edu = rec.get('education', '')
                    if rec_edu != condition['education']:
                        match = False

                # 检查技能条件
                if match and 'skills' in condition:
                    rec_skills = rec.get('skills', '')
                    if isinstance(rec_skills, str):
                        rec_skills = rec_skills.split()

                    required_skills = condition['skills']
                    if not all(skill in rec_skills for skill in required_skills):
                        match = False

                # 检查经验条件
                if match and 'experience' in condition:
                    rec_exp = rec.get('experience', 0)
                    if rec_exp < condition['experience']:
                        match = False

                # 如果匹配条件，则置顶
                if match:
                    if i > 0:  # 避免重复移动
                        reranked.insert(0, reranked.pop(i))
                    break

        # 2. 黑名单过滤
        blacklist = self.business_rules.get('blacklist', [])
        return [rec for rec in reranked if rec.get('resume_id', -1) not in blacklist]

    def rerank(self, recommendations):
        """执行重排流程"""
        if not recommendations:
            return []

        # 先应用业务规则
        recs = self.apply_business_rules(recommendations)

        # 再应用多样性规则
        recs = self.apply_diversity(recs)

        # 返回结果
        return recs[:self.diversity_rules.get('min_results', 10)]


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

        # 初始化粗排和重排模块
        self.coarse_ranker = CoarseRanker()
        self.re_ranker = ReRanker()

    def load_model_and_preprocessors(self):
        """加载模型和预处理器"""
        try:
            model_path = os.path.join(self.model_dir, 'recommendation_model.pth')
            preprocessor_path = os.path.join(self.model_dir, 'preprocessors.pkl')
            preprocessor_params_path = os.path.join(self.model_dir, 'preprocessor_params.json')

            if not os.path.exists(model_path):
                return False

            # 加载模型
            model_checkpoint = torch.load(model_path, map_location=self.device)
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

                    global tokenize_chinese
                    if 'tokenize_chinese' not in globals():
                        def tokenize_chinese(text):
                            """中文分词"""
                            if not text or text == 'NULL' or pd.isna(text):
                                return []
                            return jieba.lcut(str(text))

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

                    # 验证预处理器
                    test_text = "测试中文分词"
                    try:
                        self.skill_vectorizer.transform([test_text])
                        preprocessors_loaded = True
                        print("从JSON成功加载预处理器")
                    except Exception as e:
                        print(f"预处理器验证失败: {e}")
                except Exception as e:
                    print(f"从JSON加载预处理器失败: {e}")

            # 备选方案：直接加载pickle
            if not preprocessors_loaded and os.path.exists(preprocessor_path):
                try:
                    with open(preprocessor_path, 'rb') as f:
                        preprocessors = pickle.load(f)
                    self.skill_vectorizer = preprocessors.get('skill_vectorizer')
                    self.scaler = preprocessors.get('scaler')
                    if self.skill_vectorizer is not None and self.scaler is not None:
                        preprocessors_loaded = True
                except Exception as e:
                    print(f"从pickle加载预处理器失败: {e}")

            # 默认预处理器
            if not preprocessors_loaded:
                print("使用默认预处理器")
                self.skill_vectorizer = TfidfVectorizer(max_features=1000, tokenizer=tokenize_chinese)
                self.scaler = StandardScaler()

            return True

        except Exception as e:
            print(f"加载模型和预处理器失败: {e}")
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

        except Exception as e:
            print(f"加载简历数据失败: {e}")
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
            print(f"预计算简历特征失败: {e}")
            raise

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
            # 1. 召回阶段：基本条件筛选
            filtered_indices = self._filter_basic_requirements(job_requirements)

            if len(filtered_indices) == 0:
                print("召回阶段未找到匹配简历")
                return []

            print(f"召回阶段找到 {len(filtered_indices)} 份简历")

            # 2. 粗排阶段：轻量级模型快速筛选
            coarse_scores = []
            for idx in filtered_indices:
                resume = self.resume_data.iloc[idx]
                resume_data = {
                    'resume_id': int(resume['简历编号']),
                    'name': str(resume['姓名']),
                    'gender': str(resume['性别']),
                    'age': float(resume['年龄']) if str(resume['年龄']).replace('.', '').isdigit() else 25.0,
                    'position': str(resume['意向岗位']),
                    'education': str(resume['学历层次']),
                    'skills': str(resume['综合技能']),
                    'experience': resume.get('工作经验评分', 2.0)
                }

                try:
                    score = self.coarse_ranker.predict(resume_data, job_requirements)
                    coarse_scores.append((idx, score))
                except Exception as e:
                    print(f"粗排预测失败: {e}")
                    coarse_scores.append((idx, 0.5))  # 默认分数

            # 取粗排Top 200
            coarse_scores.sort(key=lambda x: x[1], reverse=True)
            coarse_indices = [idx for idx, _ in coarse_scores[:min(200, len(coarse_scores))]]

            if not coarse_indices:
                print("粗排阶段未找到匹配简历")
                return []

            print(f"粗排阶段筛选出 {len(coarse_indices)} 份简历")

            # 3. 精排阶段：深度学习模型精确匹配
            job_features = self._build_job_features(job_requirements)
            filtered_resume_features = self.resume_features_cache[coarse_indices]
            job_features_batch = np.tile(job_features, (len(coarse_indices), 1))

            # 模型推理
            with torch.no_grad():
                resume_tensor = torch.FloatTensor(filtered_resume_features).to(self.device)
                job_tensor = torch.FloatTensor(job_features_batch).to(self.device)
                scores = self.model(resume_tensor, job_tensor).cpu().numpy().flatten()

            # 构建推荐结果
            recommendations = []
            for i, idx in enumerate(coarse_indices):
                resume = self.resume_data.iloc[idx]
                recommendations.append({
                    'resume_id': int(resume['简历编号']),
                    'name': str(resume['姓名']),
                    'gender': str(resume['性别']),
                    'age': float(resume['年龄']) if str(resume['年龄']).replace('.', '').isdigit() else 0,
                    'position': str(resume['意向岗位']),
                    'education': str(resume['学历层次']),
                    'skills': str(resume['综合技能']),
                    'screening_result': str(resume.get('筛选结果', '')),
                    'score': float(scores[i]),  # 精排分数
                    'coarse_score': coarse_scores[i][1]  # 粗排分数
                })

            # 按精排分数排序
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            print(f"精排阶段完成，生成 {len(recommendations)} 条推荐")

            # 4. 重排阶段：业务规则和多样性调整
            recommendations = self.re_ranker.rerank(recommendations)[:top_k]
            print(f"重排阶段完成，最终推荐 {len(recommendations)} 条结果")

            return recommendations

        except Exception as e:
            print(f"推荐过程中出错: {e}")
            traceback.print_exc()
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

        except Exception as e:
            print(f"基本要求筛选失败: {e}")
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

        except Exception as e:
            print(f"搜索简历失败: {e}")
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

            # 返回所有字段
            details = {
                'resume_id': int(resume['简历编号']),
                'name': safe_get('姓名'),
                'gender': safe_get('性别'),
                'age': safe_get_int('年龄'),
                'phone': safe_get('电话'),
                'email': safe_get('邮箱'),
                'position': safe_get('意向岗位'),
                'education': safe_get('学历层次'),
                'school_type': safe_get('院校类别'),
                'major_type': safe_get('专业类别'),
                'english_level': safe_get('英语水平'),
                '编程语言': safe_get('编程语言'),
                '编程语言熟练度': safe_get('编程语言熟练度'),
                '前端技术': safe_get('前端技术'),
                '前端技术熟练度': safe_get('前端技术熟练度'),
                '后端技术': safe_get('后端技术'),
                '后端技术熟练度': safe_get('后端技术熟练度'),
                '数据库': safe_get('数据库'),
                '数据库熟练度': safe_get('数据库熟练度'),
                '云计算运维': safe_get('云计算/运维'),
                '云计算运维熟练度': safe_get('云计算/运维熟练度'),
                '数据与算法': safe_get('数据与算法'),
                '数据与算法熟练度': safe_get('数据与算法熟练度'),
                '移动开发': safe_get('移动开发'),
                '移动开发熟练度': safe_get('移动开发熟练度'),
                '测试工具': safe_get('测试工具'),
                '测试工具熟练度': safe_get('测试工具熟练度'),
                '小型企业工作经验': safe_get('小型企业工作经验'),
                '中型企业工作经验': safe_get('中型企业工作经验'),
                '大型企业工作经验': safe_get('大型企业工作经验'),
                '小规模项目': safe_get_int('小规模项目'),
                '中规模项目': safe_get_int('中规模项目'),
                '大规模项目': safe_get_int('大规模项目'),
                'screening_result': safe_get('筛选结果')
            }
            return details

        except Exception as e:
            print(f"获取简历详情失败: {e}")
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

        except Exception as e:
            print(f"获取统计信息失败: {e}")
            return {}

    def initialize(self, data_path='data/Chinese_resume_data.csv'):
        """初始化推理引擎"""
        if not self.load_model_and_preprocessors():
            print("模型和预处理器加载失败")
            return False
        if not self.load_resume_data(data_path):
            print("简历数据加载失败")
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
    # 测试推理引擎
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
        print("\n最终推荐结果:")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['name']} - 职位: {rec['position']}, 精排: {rec['score']:.4f}, 粗排: {rec['coarse_score']:.4f}")
    else:
        print("推理引擎初始化失败")
