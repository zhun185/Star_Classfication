<template>
  <q-page padding>
    <h4 class="q-my-md">分类预测</h4>
    <p v-if="!loadedSpectrumPath && !loadedImagePath">上传光谱和图像文件，选择模型进行分类预测。</p>
    <p v-else>正在使用从数据管理页面传入的光谱和图像文件进行预测。您也可以 <a href="javascript:void(0)" @click="clearLoadedPathsAndFiles">手动上传新文件</a>。</p>

    <q-card>
      <q-card-section>
        <div class="text-h6">输入数据</div>
      </q-card-section>
      <q-card-section class="q-gutter-md">
        <q-select
          v-model="selectedModel"
          :options="modelOptions"
          label="选择模型"
          filled
          emit-value
          map-options
          :loading="isLoadingModels"
          :disable="isLoadingModels || isPredicting"
          hint="从后端加载可用模型"
        />
        <div v-if="loadedSpectrumPath" class="q-mb-md">
          <q-banner inline-actions rounded class="bg-grey-2 text-grey-8">
            <template v-slot:avatar>
              <q-icon name="mdi-check-circle" color="positive" />
            </template>
            光谱文件 (来自路径): <strong>{{ loadedSpectrumPath }}</strong>
            <template v-slot:action>
              <q-btn flat dense round icon="mdi-close-circle" @click="clearLoadedSpectrumFile" title="清除此文件并手动上传" v-if="!isPredicting"/>
            </template>
          </q-banner>
        </div>
        <q-file v-else v-model="spectrumFile" label="上传光谱文件 (.fits, .fit, .fts)" accept=".fits,.fit,.fts" filled counter :disable="isPredicting">
          <template v-slot:prepend>
            <q-icon name="mdi-chart-scatter-plot" />
          </template>
        </q-file>

        <div v-if="loadedImagePath" class="q-mb-md">
           <q-banner inline-actions rounded class="bg-grey-2 text-grey-8">
            <template v-slot:avatar>
              <q-icon name="mdi-check-circle" color="positive" />
            </template>
            图像文件 (来自路径): <strong>{{ loadedImagePath }}</strong>
            <template v-slot:action>
              <q-btn flat dense round icon="mdi-close-circle" @click="clearLoadedImageFile" title="清除此文件并手动上传" v-if="!isPredicting"/>
            </template>
          </q-banner>
        </div>
        <q-file v-else v-model="imageFile" label="上传图像文件 (.jpg, .png, .jpeg)" accept=".jpg,.jpeg,.png" filled counter :disable="isPredicting">
          <template v-slot:prepend>
            <q-icon name="mdi-image" />
          </template>
        </q-file>
      </q-card-section>
      <q-card-actions align="right">
        <q-btn label="开始预测" color="primary" @click="startPrediction" :loading="isPredicting" :disable="!selectedModel || !spectrumFile || !imageFile" />
      </q-card-actions>
    </q-card>

    <q-card class="q-mt-md" v-if="predictionResult && !isPredicting">
      <q-card-section>
        <div class="text-h6">预测结果</div>
      </q-card-section>
      <q-card-section>
        <p><strong>预测类别:</strong> <q-chip :label="predictionResult.class" color="secondary" text-color="white" size="lg" clickable @click="copyToClipboard(predictionResult.class)"/></p>
        <p><strong>置信度:</strong> {{ predictionResult.confidence.toFixed(2) }} %</p>
      </q-card-section>
    </q-card>

    <q-banner v-if="predictionError && !isPredicting" class="text-white bg-red q-mt-md" rounded>
      <template v-slot:avatar>
        <q-icon name="error" />
      </template>
      预测失败: {{ predictionError }}
    </q-banner>

  </q-page>
</template>

<script lang="ts" setup>
import { ref, onMounted, watch } from 'vue';
import { useRoute } from 'vue-router';
import { useQuasar, QSelectOption } from 'quasar';
import axios from 'axios'; // 保持 axios 用于 POST predict 和新的 GET fetch_file

interface PredictionResult {
  class: string; // Changed from type to class to match backend
  confidence: number;
}

const $q = useQuasar();
const route = useRoute();
const spectrumFile = ref<File | null>(null);
const imageFile = ref<File | null>(null);
const selectedModel = ref<string | null>(null);
const modelOptions = ref<QSelectOption<string>[]>([]); // For q-select: {label: string, value: string}
const isLoadingModels = ref(false);

const predictionResult = ref<PredictionResult | null>(null);
const predictionError = ref<string | null>(null);
const isPredicting = ref(false);

const API_BASE_URL = 'http://localhost:8000';
// const SPECTRA_BASE_URL = `${API_BASE_URL}/static/spectra`; // 不再需要这个
// const IMAGES_BASE_URL = `${API_BASE_URL}/static/images`;   // 不再需要这个

// 用于显示从路径加载的文件名
const loadedSpectrumPath = ref<string | null>(null); // 新增
const loadedImagePath = ref<string | null>(null);    // 新增

const fetchModels = async () => {
  isLoadingModels.value = true;
  try {
    const response = await axios.get<string[]>(`${API_BASE_URL}/api/models`);
    modelOptions.value = response.data.map(modelName => ({ label: modelName, value: modelName }));
    if (modelOptions.value.length > 0) {
        // selectedModel.value = modelOptions.value[0].value; // Optionally pre-select first model
        $q.notify({ message: '模型列表加载成功', color: 'positive', icon: 'check_circle' });
    } else {
        $q.notify({ message: '未能从后端加载到可用模型。', color: 'warning', icon: 'warning' });
    }
  } catch (error) {
    console.error('Error fetching models:', error);
    $q.notify({ type: 'negative', message: '加载模型列表失败，请检查后端服务是否运行。', multiLine: true });
    modelOptions.value = []; // Clear options on error
  }
  isLoadingModels.value = false;
};

// 修改：根据相对路径和类型，通过新的API端点获取文件对象
async function fetchFileFromPath(relativePath: string, fileType: 'spectrum' | 'image'): Promise<File | null> {
  if (!relativePath) return null;
  const defaultFileName = relativePath.split('/').pop() || (fileType === 'spectrum' ? 'spectrum.fits' : 'image.jpg');
  $q.loading.show({ message: `正在加载 ${defaultFileName} (API)...` });
  
  try {
    const response = await axios.get(`${API_BASE_URL}/api/fetch_file/${fileType}/${relativePath}`, {
      responseType: 'blob', // 重要：期望响应是一个Blob
    });

    // Axios 成功状态码通常是 2xx
    if (response.status < 200 || response.status >= 300) {
      throw new Error(`下载文件失败: ${response.status} ${response.statusText}`);
    }
    
    const blob = response.data as Blob;
    
    let mimeType = 'application/octet-stream'; // 默认MIME类型
    if (fileType === 'spectrum' && (defaultFileName.endsWith('.fits') || defaultFileName.endsWith('.fit') || defaultFileName.endsWith('.fts'))) {
      mimeType = 'application/fits'; 
    } else if (fileType === 'image') {
      if (defaultFileName.endsWith('.jpg') || defaultFileName.endsWith('.jpeg')) {
        mimeType = 'image/jpeg';
      } else if (defaultFileName.endsWith('.png')) {
        mimeType = 'image/png';
      }
      // 可以根据需要添加更多图像类型
    }

    return new File([blob], defaultFileName, { type: mimeType });
  } catch (error) {
    console.error(`Error fetching file ${relativePath} (type: ${fileType}) via API:`, error);
    let errorMessage = `加载 ${defaultFileName} (API) 失败.`;
    if (axios.isAxiosError(error)) {
      if (error.response) {
        // 尝试从 error.response.data (如果是JSON错误对象) 获取更具体的后端错误信息
        const serverError = error.response.data?.detail || error.response.statusText || error.message;
        errorMessage += ` 错误: ${error.response.status} - ${serverError}`;
      } else if (error.request) {
        errorMessage += ' 网络请求错误，无法连接到服务器。';
      } else {
        errorMessage += ` 错误: ${error.message}`;
      }
    } else if (error instanceof Error) {
        errorMessage += ` 错误: ${error.message}`;
    } else {
        errorMessage += ' 未知错误.';
    }
    $q.notify({ type: 'negative', message: errorMessage, multiLine: true, timeout: 7000 });
    return null;
  } finally {
    $q.loading.hide();
  }
}

const startPrediction = async () => {
  console.log('Spectrum file to be sent:', spectrumFile.value);
  console.log('Image file to be sent:', imageFile.value);
  console.log('Selected model to be sent:', selectedModel.value);

  if (!spectrumFile.value || !imageFile.value || !selectedModel.value) {
    $q.notify({ type: 'negative', message: '请上传光谱文件、图像文件并选择一个模型。' });
    return;
  }

  isPredicting.value = true;
  predictionResult.value = null;
  predictionError.value = null;

  const formData = new FormData();
  formData.append('spectrum_file', spectrumFile.value);
  formData.append('image_file', imageFile.value);
  formData.append('model_name', selectedModel.value);

  try {
    const response = await axios.post<any>(`${API_BASE_URL}/api/predict`, formData, { 
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    if (response.data && response.data.class !== undefined) {
      predictionResult.value = response.data as PredictionResult; 
      $q.notify({ message: '预测成功!', color: 'positive', icon: 'check_circle'});
    } else {
      console.warn("Received unknown prediction response structure during final prediction:", response.data);
      $q.notify({ message: '收到未知的预测响应结构。', color: 'warning', icon: 'warning' });
      predictionResult.value = null;
    }

  } catch (error: any) {
    console.error('Prediction error:', error);
    if (error.response) {
      predictionError.value = `错误 ${error.response.status}: ${error.response.data.detail || error.response.statusText}`;
    } else if (error.request) {
      predictionError.value = '无法连接到服务器，请检查后端服务是否运行以及网络连接。';
    } else {
      predictionError.value = error.message || '发生未知错误';
    }
    $q.notify({ type: 'negative', message: predictionError.value, multiLine: true, timeout: 7000 });
  }
  isPredicting.value = false;
};

const copyToClipboard = (text: string) => {
  navigator.clipboard.writeText(text).then(() => {
    $q.notify({ message: `'${text}' 已复制到剪贴板`, color: 'info', icon: 'content_copy', position: 'bottom', timeout: 2000 });
  }).catch(err => {
    console.error('Failed to copy: ', err);
    $q.notify({ message: '复制失败', color: 'negative'});
  });
};

const processRouteParams = async () => {
  const specPath = route.query.spectrumPath as string | undefined;
  const imgPath = route.query.imagePath as string | undefined;

  if (specPath && imgPath) {
    $q.notify({ message: '检测到传入路径，尝试通过API加载文件...', icon: 'info', color: 'info', multiLine: true, position: 'top' });

    if (modelOptions.value.length === 0) { await fetchModels(); }
    if (modelOptions.value.length > 0 && !selectedModel.value) {
      selectedModel.value = modelOptions.value[0].value;
      console.log(`已自动选择模型: ${selectedModel.value}`);
    }

    loadedSpectrumPath.value = specPath;
    loadedImagePath.value = imgPath;
    spectrumFile.value = null; 
    imageFile.value = null;

    const fetchedSpectrumFile = await fetchFileFromPath(specPath, 'spectrum');
    const fetchedImageFile = await fetchFileFromPath(imgPath, 'image');

    if (fetchedSpectrumFile && fetchedImageFile) {
      spectrumFile.value = fetchedSpectrumFile;
      imageFile.value = fetchedImageFile;
      if (selectedModel.value) {
        await startPrediction();
      } else {
        $q.notify({ type: 'warning', message: '文件已加载，但模型未选。请选择模型后手动预测。', multiLine: true });
      }
    } else {
      $q.notify({ type: 'negative', message: '无法通过API加载一个或两个文件。请检查路径或手动上传。', multiLine: true });
      loadedSpectrumPath.value = null;
      loadedImagePath.value = null;
    }
  }
};

onMounted(async () => {
  await fetchModels(); // 先加载模型
  await processRouteParams(); // 然后处理路由参数
});

// 新增: 监听路由变化，以便在参数变化时重新处理
watch(() => route.query, async (newQuery, oldQuery) => {
  if (newQuery.spectrumPath !== oldQuery.spectrumPath || newQuery.imagePath !== oldQuery.imagePath) {
    // 清理旧状态，准备重新加载
    predictionResult.value = null;
    predictionError.value = null;
    spectrumFile.value = null;
    imageFile.value = null;
    loadedSpectrumPath.value = null;
    loadedImagePath.value = null;
    await processRouteParams();
  }
}, { deep: true });

const clearLoadedPathsAndFiles = () => {
  loadedSpectrumPath.value = null;
  loadedImagePath.value = null;
  spectrumFile.value = null;
  imageFile.value = null;
};

const clearLoadedSpectrumFile = () => {
  loadedSpectrumPath.value = null;
  spectrumFile.value = null;
};

const clearLoadedImageFile = () => {
  loadedImagePath.value = null;
  imageFile.value = null;
};

</script>

<style scoped>
.q-card {
  max-width: 600px;
  margin: 20px auto;
}
.q-banner {
  max-width: 600px;
  margin: 20px auto;
}
</style> 