"""
简历推荐系统训练模块
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import jieba
import pickle
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def tokenize_chinese(text):
    """中文分词"""
    if not text or text == 'NULL' or pd.isna(text):
        return []
    return jieba.lcut(str(text))


class ResumeDataset(Dataset):
    """简历数据集"""

    def __init__(self, resume_features, job_features, labels):
        self.resume_features = torch.FloatTensor(resume_features)
        self.job_features = torch.FloatTensor(job_features)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'resume_features': self.resume_features[idx],
            'job_features': self.job_features[idx],
            'label': self.labels[idx]
        }


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


class ResumeRecommendationTrainer:
    """推荐系统训练器"""

    def __init__(self, data_path='data/Chinese_resume_data.csv'):
        self.data_path = data_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.resume_data = None
        self.skill_vectorizer = TfidfVectorizer(max_features=1000, tokenizer=tokenize_chinese)
        self.scaler = StandardScaler()
        self.model = None
        self.training_history = {'train_loss': [], 'val_loss': [], 'metrics': []}

    def load_and_preprocess_data(self):
        """加载和预处理数据"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"数据文件不存在: {self.data_path}")

        try:
            self.resume_data = pd.read_csv(self.data_path, encoding='utf-8')
        except UnicodeDecodeError:
            self.resume_data = pd.read_csv(self.data_path, encoding='gbk')

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
        self.resume_data['学历评分'] = self.resume_data['学历层次'].map(education_mapping).fillna(2)

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
                    multiplier = 1.5 if exp_type == '大型企业工作经验' else 1.2 if exp_type == '中型企业工作经验' else 1.0
                    score += exp_mapping.get(exp_value, 0) * multiplier
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

    def prepare_training_data(self):
        """准备训练数据"""
        # 训练TF-IDF向量器
        skill_vectors = self.skill_vectorizer.fit_transform(self.resume_data['综合技能']).toarray()

        # 构建简历特征
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

        resume_features = np.array(resume_features)

        # 构建训练数据
        job_features = []
        labels = []
        resume_indices = []

        positions = self.resume_data['意向岗位'].unique()

        for position in positions:
            position_data = self.resume_data[self.resume_data['意向岗位'] == position]
            if len(position_data) < 2:
                continue

            # 构建职位特征
            pos_skill_text = ' '.join(position_data['综合技能'].tolist())
            pos_skill_vector = self.skill_vectorizer.transform([pos_skill_text]).toarray()[0]

            avg_education = position_data['学历评分'].mean()
            avg_experience = position_data['工作经验评分'].mean()
            avg_age = pd.to_numeric(position_data['年龄'], errors='coerce').mean()

            pos_features = list(pos_skill_vector) + [avg_education, avg_experience, avg_age]

            for idx in position_data.index:
                job_features.append(pos_features)
                resume_indices.append(idx)

                screening_result = self.resume_data.loc[idx, '筛选结果']
                comprehensive_score = self._calculate_comprehensive_score(self.resume_data.loc[idx])

                if screening_result == '建议面试':
                    labels.append(1.0)
                elif screening_result == '不建议面试':
                    labels.append(0.0)
                else:
                    labels.append(1.0 if comprehensive_score > 2.5 else 0.0)

        selected_resume_features = resume_features[resume_indices]
        selected_resume_features = self.scaler.fit_transform(selected_resume_features)
        job_features = np.array(job_features)
        labels = np.array(labels)

        # 数据平衡处理
        positive_ratio = np.mean(labels)
        if positive_ratio < 0.3 or positive_ratio > 0.7:
            labels = self._balance_labels(labels)

        return selected_resume_features, job_features, labels

    def _balance_labels(self, labels):
        """平衡标签分布"""
        positive_ratio = np.mean(labels)

        if positive_ratio < 0.3:
            negative_indices = np.where(labels == 0)[0]
            n_to_flip = min(len(negative_indices) // 3, len(negative_indices))
            flip_indices = np.random.choice(negative_indices, n_to_flip, replace=False)
            labels[flip_indices] = 1.0
        elif positive_ratio > 0.7:
            positive_indices = np.where(labels == 1)[0]
            n_to_flip = min(len(positive_indices) // 3, len(positive_indices))
            flip_indices = np.random.choice(positive_indices, n_to_flip, replace=False)
            labels[flip_indices] = 0.0

        return labels

    def _calculate_comprehensive_score(self, resume):
        """计算综合评分"""
        score = 0
        score += resume.get('学历评分', 2) * 0.2
        score += min(resume.get('工作经验评分', 2) / 6, 1) * 0.3
        score += min(resume.get('项目经验评分', 2) / 15, 1) * 0.25

        skills = str(resume.get('综合技能', ''))
        skill_count = len([skill for skill in skills.split() if skill != 'NULL' and skill])
        score += min(skill_count / 8, 1) * 0.25

        return score * 4

    def create_model(self, resume_feature_dim, job_feature_dim, use_sigmoid=True):
        """创建模型"""
        self.model = DeepRecommendationModel(
            resume_feature_dim=resume_feature_dim,
            job_feature_dim=job_feature_dim,
            use_sigmoid=use_sigmoid
        ).to(self.device)

    def train_model(self, resume_features, job_features, labels, epochs=100, batch_size=64, learning_rate=0.001):
        """训练模型"""
        X_res_train, X_res_val, X_job_train, X_job_val, y_train, y_val = train_test_split(
            resume_features, job_features, labels, test_size=0.2, random_state=42, stratify=labels
        )

        train_dataset = ResumeDataset(X_res_train, X_job_train, y_train)
        val_dataset = ResumeDataset(X_res_val, X_job_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        pos_weight = torch.FloatTensor([class_weights[1] / class_weights[0]]).to(self.device)

        if self.model.use_sigmoid:
            criterion = nn.BCELoss()
        else:
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=15, factor=0.5)

        best_val_loss = float('inf')
        patience_counter = 0
        best_f1 = 0.0

        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_predictions = []
            train_targets = []

            for batch in train_loader:
                resume_feat = batch['resume_features'].to(self.device)
                job_feat = batch['job_features'].to(self.device)
                target = batch['label'].to(self.device).unsqueeze(1)

                optimizer.zero_grad()
                output = self.model(resume_feat, job_feat)
                loss = criterion(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()

                if self.model.use_sigmoid:
                    predictions = output
                else:
                    predictions = torch.sigmoid(output)

                train_predictions.extend((predictions > 0.5).cpu().numpy().flatten())
                train_targets.extend(target.cpu().numpy().flatten())

            train_loss /= len(train_loader)

            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            val_predictions = []
            val_targets = []

            with torch.no_grad():
                for batch in val_loader:
                    resume_feat = batch['resume_features'].to(self.device)
                    job_feat = batch['job_features'].to(self.device)
                    target = batch['label'].to(self.device).unsqueeze(1)

                    output = self.model(resume_feat, job_feat)
                    loss = criterion(output, target)
                    val_loss += loss.item()

                    if self.model.use_sigmoid:
                        predictions = output
                    else:
                        predictions = torch.sigmoid(output)

                    val_predictions.extend((predictions > 0.5).cpu().numpy().flatten())
                    val_targets.extend(target.cpu().numpy().flatten())

            val_loss /= len(val_loader)
            scheduler.step(val_loss)

            # 计算指标
            train_f1 = f1_score(train_targets, train_predictions, zero_division=0)
            val_f1 = f1_score(val_targets, val_predictions, zero_division=0)
            val_precision = precision_score(val_targets, val_predictions, zero_division=0)
            val_recall = recall_score(val_targets, val_predictions, zero_division=0)

            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['metrics'].append({
                'train_f1': train_f1,
                'val_f1': val_f1,
                'val_precision': val_precision,
                'val_recall': val_recall
            })

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}')

            # 早停机制
            if val_f1 > best_f1 or (val_f1 == best_f1 and val_loss < best_val_loss):
                best_val_loss = val_loss
                best_f1 = val_f1
                patience_counter = 0

                os.makedirs('models', exist_ok=True)
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_val_loss': best_val_loss,
                    'best_f1': best_f1,
                    'use_sigmoid': self.model.use_sigmoid,
                    'model_config': {
                        'resume_feature_dim': self.model.resume_feature_dim,
                        'job_feature_dim': self.model.job_feature_dim,
                        'hidden_dims': [512, 256, 128],
                        'use_sigmoid': self.model.use_sigmoid
                    }
                }, 'models/best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= 30:
                    break

    def save_model_and_artifacts(self):
        """保存模型和相关组件"""
        os.makedirs('models', exist_ok=True)

        # 加载最佳模型
        if os.path.exists('models/best_model.pth'):
            checkpoint = torch.load('models/best_model.pth', map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])

        # 保存PyTorch模型
        model_save_dict = {
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'resume_feature_dim': self.model.resume_feature_dim,
                'job_feature_dim': self.model.job_feature_dim,
                'hidden_dims': [512, 256, 128],
                'use_sigmoid': self.model.use_sigmoid
            },
            'model_type': 'DeepRecommendationModel',
            'pytorch_version': torch.__version__,
        }

        torch.save(model_save_dict, 'models/recommendation_model.pth')

        # 保存预处理器
        pickle_success = False

        # 尝试pickle保存
        try:
            preprocessor_dict = {
                'skill_vectorizer': self.skill_vectorizer,
                'scaler': self.scaler,
            }
            with open('models/preprocessors.pkl', 'wb') as f:
                pickle.dump(preprocessor_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
            pickle_success = True
        except Exception:
            pass

        # JSON参数备份
        try:
            vectorizer_params = {
                'vocabulary_': dict(self.skill_vectorizer.vocabulary_) if hasattr(self.skill_vectorizer, 'vocabulary_') else {},
                'idf_': self.skill_vectorizer.idf_.tolist() if hasattr(self.skill_vectorizer, 'idf_') else [],
                'max_features': getattr(self.skill_vectorizer, 'max_features', 1000),
            }

            scaler_params = {
                'mean_': self.scaler.mean_.tolist() if hasattr(self.scaler, 'mean_') else [],
                'scale_': self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') else [],
                'var_': self.scaler.var_.tolist() if hasattr(self.scaler, 'var_') else [],
                'n_features_in_': getattr(self.scaler, 'n_features_in_', 0),
                'n_samples_seen_': getattr(self.scaler, 'n_samples_seen_', 0),
            }

            preprocessor_params = {
                'vectorizer_params': vectorizer_params,
                'scaler_params': scaler_params,
            }

            with open('models/preprocessor_params.json', 'w', encoding='utf-8') as f:
                json.dump(preprocessor_params, f, indent=2, ensure_ascii=False)

        except Exception:
            pass

        # 保存训练历史
        try:
            with open('models/training_history.json', 'w', encoding='utf-8') as f:
                json.dump(self.training_history, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

        # 保存元数据
        try:
            metadata = {
                'training_date': datetime.now().isoformat(),
                'model_type': 'DeepRecommendationModel',
                'framework': 'PyTorch',
                'pytorch_version': torch.__version__,
                'data_shape': {
                    'n_samples': len(self.resume_data),
                    'resume_features': self.model.resume_feature_dim,
                    'job_features': self.model.job_feature_dim
                },
                'training_params': {
                    'device': str(self.device),
                    'epochs_trained': len(self.training_history['train_loss']),
                    'hidden_dims': [512, 256, 128],
                    'use_sigmoid': self.model.use_sigmoid,
                },
                'preprocessing': {
                    'pickle_saved': pickle_success,
                    'json_backup_saved': True,
                    'tokenizer_function': 'tokenize_chinese'
                }
            }

            with open('models/metadata.json', 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def run_training_pipeline(self):
        """运行完整训练流程"""
        try:
            os.makedirs('models', exist_ok=True)

            self.load_and_preprocess_data()
            resume_features, job_features, labels = self.prepare_training_data()

            self.create_model(
                resume_feature_dim=resume_features.shape[1],
                job_feature_dim=job_features.shape[1],
                use_sigmoid=True
            )

            self.train_model(resume_features, job_features, labels)
            self.save_model_and_artifacts()

            return True

        except Exception as e:
            print(f"训练失败: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    trainer = ResumeRecommendationTrainer()
    success = trainer.run_training_pipeline()

    if success:
        print("训练成功完成")
    else:
        print("训练失败")