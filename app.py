"""
简历推荐系统后端API服务
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
from datetime import datetime
from inference_engine import get_inference_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# 全局变量
engine_loaded = False
inference_engine = None
first_request_done = False


@app.before_request
def initialize():
    """初始化推理引擎"""
    global engine_loaded, inference_engine, first_request_done

    if not first_request_done:
        first_request_done = True
        try:
            if os.path.exists('models/recommendation_model.pth'):
                inference_engine = get_inference_engine()
                engine_loaded = True
                logger.info("推理引擎初始化成功")
            else:
                logger.warning("未找到训练好的模型文件，请先运行训练流程")
        except Exception as e:
            logger.error(f"推理引擎初始化失败: {str(e)}")


@app.route('/')
def home():
    """首页"""
    try:
        with open('index.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return jsonify({
            'message': '简历推荐系统API',
            'version': '2.0.0',
            'status': 'running',
            'engine_loaded': engine_loaded,
            'timestamp': datetime.now().isoformat()
        })


@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        'status': 'healthy' if engine_loaded else 'model_not_loaded',
        'engine_loaded': engine_loaded,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """获取数据统计"""
    if not engine_loaded:
        return jsonify({'error': '推理引擎未加载'}), 400

    try:
        stats = inference_engine.get_statistics()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"获取统计信息失败: {str(e)}")
        return jsonify({'error': '获取统计信息失败'}), 500


@app.route('/api/positions', methods=['GET'])
def get_positions():
    """获取职位列表"""
    if not engine_loaded:
        return jsonify({'error': '推理引擎未加载'}), 400

    try:
        stats = inference_engine.get_statistics()
        positions = list(stats.get('positions', {}).keys())
        return jsonify({'positions': positions})
    except Exception as e:
        logger.error(f"获取职位列表失败: {str(e)}")
        return jsonify({'error': '获取职位列表失败'}), 500


@app.route('/api/recommend/deep', methods=['POST'])
def recommend_deep_learning():
    """深度学习推荐"""
    if not engine_loaded:
        return jsonify({'error': '推理引擎未加载'}), 400

    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': '请求数据为空'}), 400

        job_requirements = {
            'position': data.get('position', ''),
            'required_skills': data.get('skills', ''),
            'min_education': data.get('min_education', ''),
            'min_experience_years': data.get('min_experience', 0),
            'max_age': data.get('max_age', 100),
            'min_age': data.get('min_age', 18)
        }

        top_k = min(data.get('top_k', 10), 50)
        recommendations = inference_engine.recommend(job_requirements, top_k)

        return jsonify({
            'status': 'success',
            'method': 'deep_learning',
            'job_requirements': job_requirements,
            'recommendations': recommendations,
            'total_count': len(recommendations)
        })

    except Exception as e:
        logger.error(f"深度学习推荐失败: {str(e)}")
        return jsonify({'error': f'推荐失败: {str(e)}'}), 500


@app.route('/api/recommend/content', methods=['POST'])
def recommend_content_based():
    """基于内容的推荐(兼容)"""
    return recommend_deep_learning()


@app.route('/api/recommend/collaborative', methods=['POST'])
def recommend_collaborative():
    """协同过滤推荐(兼容)"""
    return recommend_deep_learning()


@app.route('/api/recommend/hybrid', methods=['POST'])
def recommend_hybrid():
    """混合推荐(兼容)"""
    return recommend_deep_learning()


@app.route('/api/resume/<int:resume_id>', methods=['GET'])
def get_resume_details(resume_id):
    """获取简历详情"""
    if not engine_loaded:
        return jsonify({'error': '推理引擎未加载'}), 400

    try:
        resume_details = inference_engine.get_resume_details(resume_id)
        if not resume_details:
            return jsonify({'error': '简历不存在'}), 404

        return jsonify({
            'status': 'success',
            'resume': resume_details
        })
    except Exception as e:
        logger.error(f"获取简历详情失败: {str(e)}")
        return jsonify({'error': '获取简历详情失败'}), 500


@app.route('/api/search', methods=['POST'])
def search_resumes():
    """搜索简历"""
    if not engine_loaded:
        return jsonify({'error': '推理引擎未加载'}), 400

    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': '请求数据为空'}), 400

        search_params = {
            'position': data.get('position'),
            'education': data.get('education'),
            'gender': data.get('gender'),
            'min_age': data.get('min_age'),
            'max_age': data.get('max_age'),
            'skills': data.get('skills'),
            'limit': data.get('limit', 20)
        }

        results = inference_engine.search_resumes(search_params)

        return jsonify({
            'status': 'success',
            'total_count': len(results),
            'resumes': results
        })
    except Exception as e:
        logger.error(f"搜索简历失败: {str(e)}")
        return jsonify({'error': '搜索失败'}), 500


@app.route('/api/model/info', methods=['GET'])
def get_model_info():
    """获取模型信息"""
    try:
        metadata = {}
        training_history = {}

        if os.path.exists('models/metadata.json'):
            import json
            with open('models/metadata.json', 'r') as f:
                metadata = json.load(f)

        if os.path.exists('models/training_history.json'):
            with open('models/training_history.json', 'r') as f:
                training_history = json.load(f)

        model_info = {
            'status': 'loaded' if engine_loaded else 'not_loaded',
            'metadata': metadata,
            'training_history': {
                'epochs_trained': len(training_history.get('train_loss', [])),
                'final_train_loss': training_history.get('train_loss', [])[-1] if training_history.get('train_loss') else None,
                'final_val_loss': training_history.get('val_loss', [])[-1] if training_history.get('val_loss') else None,
                'best_metrics': training_history.get('metrics', [])[-1] if training_history.get('metrics') else None
            }
        }

        return jsonify(model_info)
    except Exception as e:
        logger.error(f"获取模型信息失败: {str(e)}")
        return jsonify({'error': '获取模型信息失败'}), 500


@app.route('/api/upload', methods=['POST'])
def upload_data():
    """上传数据文件"""
    if 'file' not in request.files:
        return jsonify({'error': '没有文件'}), 400

    file = request.files['file']
    if file.filename == '' or not file.filename.endswith('.csv'):
        return jsonify({'error': '请上传CSV文件'}), 400

    try:
        os.makedirs('data', exist_ok=True)
        filepath = os.path.join('data', 'Chinese_resume_data.csv')
        file.save(filepath)

        return jsonify({
            'status': 'success',
            'message': '数据文件上传成功，请重新训练模型'
        })
    except Exception as e:
        logger.error(f"上传数据失败: {str(e)}")
        return jsonify({'error': f'上传失败: {str(e)}'}), 500


@app.route('/api/retrain', methods=['POST'])
def trigger_retrain():
    """触发重新训练"""
    return jsonify({
        'message': '要重新训练模型，请运行：python train.py'
    })


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': '接口不存在'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': '服务器内部错误'}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'

    print("启动简历推荐系统API服务...")
    print(f"访问地址: http://localhost:{port}")

    app.run(host='0.0.0.0', port=port, debug=debug)