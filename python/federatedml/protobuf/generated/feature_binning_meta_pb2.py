# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: feature-binning-meta.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1a\x66\x65\x61ture-binning-meta.proto\x12&com.webank.ai.fate.core.mlmodel.buffer\"?\n\rTransformMeta\x12\x16\n\x0etransform_cols\x18\x01 \x03(\x03\x12\x16\n\x0etransform_type\x18\x02 \x01(\t\"\xc2\x02\n\x12\x46\x65\x61tureBinningMeta\x12\x10\n\x08need_run\x18\x01 \x01(\x08\x12\x0e\n\x06method\x18\n \x01(\t\x12\x16\n\x0e\x63ompress_thres\x18\x02 \x01(\x03\x12\x11\n\thead_size\x18\x03 \x01(\x03\x12\r\n\x05\x65rror\x18\x04 \x01(\x01\x12\x0f\n\x07\x62in_num\x18\x05 \x01(\x03\x12\x0c\n\x04\x63ols\x18\x06 \x03(\t\x12\x19\n\x11\x61\x64justment_factor\x18\x07 \x01(\x01\x12\x12\n\nlocal_only\x18\x08 \x01(\x08\x12N\n\x0ftransform_param\x18\t \x01(\x0b\x32\x35.com.webank.ai.fate.core.mlmodel.buffer.TransformMeta\x12\x13\n\x0bskip_static\x18\x0b \x01(\x08\x12\x1d\n\x15optimal_metric_method\x18\x0c \x01(\tB\x19\x42\x17\x46\x65\x61tureBinningMetaProtob\x06proto3')



_TRANSFORMMETA = DESCRIPTOR.message_types_by_name['TransformMeta']
_FEATUREBINNINGMETA = DESCRIPTOR.message_types_by_name['FeatureBinningMeta']
TransformMeta = _reflection.GeneratedProtocolMessageType('TransformMeta', (_message.Message,), {
  'DESCRIPTOR' : _TRANSFORMMETA,
  '__module__' : 'feature_binning_meta_pb2'
  # @@protoc_insertion_point(class_scope:com.webank.ai.fate.core.mlmodel.buffer.TransformMeta)
  })
_sym_db.RegisterMessage(TransformMeta)

FeatureBinningMeta = _reflection.GeneratedProtocolMessageType('FeatureBinningMeta', (_message.Message,), {
  'DESCRIPTOR' : _FEATUREBINNINGMETA,
  '__module__' : 'feature_binning_meta_pb2'
  # @@protoc_insertion_point(class_scope:com.webank.ai.fate.core.mlmodel.buffer.FeatureBinningMeta)
  })
_sym_db.RegisterMessage(FeatureBinningMeta)

if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'B\027FeatureBinningMetaProto'
  _TRANSFORMMETA._serialized_start=70
  _TRANSFORMMETA._serialized_end=133
  _FEATUREBINNINGMETA._serialized_start=136
  _FEATUREBINNINGMETA._serialized_end=458
# @@protoc_insertion_point(module_scope)
