  *	?????Q?@2T
Iterator::Prefetch::GeneratorI.?!????!?>?2T@)I.?!????1?>?2T@:Preprocessing2?
IIterator::Model::Prefetch::Rebatch::BatchV2::MemoryCacheImpl::TensorSlice@k?w??#??!sㅿ?<@)k?w??#??1sㅿ?<@:Preprocessing2s
<Iterator::Model::Prefetch::Rebatch::BatchV2::MemoryCacheImpl4?[ A???!]?pC$@)????<,??1rٍz@@:Preprocessing2o
8Iterator::Model::Prefetch::Rebatch::BatchV2::MemoryCache@?v??/??!?t?bL-@)]m???{??1??Ҁ>?@:Preprocessing2F
Iterator::Model?A`??"??!????
>@)8??d?`??1?}u@:Preprocessing2P
Iterator::Model::PrefetchF%u?{?!????#??)F%u?{?1????#??:Preprocessing2I
Iterator::Prefetch-C??6j?!???90Q??)-C??6j?1???90Q??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisg
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*no#You may skip the rest of this page.BX
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.