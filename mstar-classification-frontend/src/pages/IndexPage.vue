<template>
  <q-page class="row items-center justify-evenly">
    <div class="text-center">
      <h1 class="text-h3">M-Star 分类系统</h1>
      <p class="text-subtitle1">欢迎使用基于光谱和图像数据的M型星分类工具。</p>
      <q-btn color="primary" label="开始探索" to="/data" />
    </div>
  </q-page>
</template>

<script lang="ts">
import { defineComponent } from 'vue';

export default defineComponent({
  name: 'IndexPage',
});
</script> 