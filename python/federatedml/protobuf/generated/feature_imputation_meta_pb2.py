# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: feature-imputation-meta.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1d\x66\x65\x61ture-imputation-meta.proto\x12&com.webank.ai.fate.core.mlmodel.buffer\"\x87\x02\n\x12\x46\x65\x61tureImputerMeta\x12\x12\n\nis_imputer\x18\x01 \x01(\x08\x12\x10\n\x08strategy\x18\x02 \x01(\t\x12\x15\n\rmissing_value\x18\x03 \x03(\t\x12\x1a\n\x12missing_value_type\x18\x04 \x03(\t\x12\x63\n\rcols_strategy\x18\x05 \x03(\x0b\x32L.com.webank.ai.fate.core.mlmodel.buffer.FeatureImputerMeta.ColsStrategyEntry\x1a\x33\n\x11\x43olsStrategyEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"{\n\x15\x46\x65\x61tureImputationMeta\x12P\n\x0cimputer_meta\x18\x01 \x01(\x0b\x32:.com.webank.ai.fate.core.mlmodel.buffer.FeatureImputerMeta\x12\x10\n\x08need_run\x18\x02 \x01(\x08\x42\x1c\x42\x1a\x46\x65\x61tureImputationMetaProtob\x06proto3')



_FEATUREIMPUTERMETA = DESCRIPTOR.message_types_by_name['FeatureImputerMeta']
_FEATUREIMPUTERMETA_COLSSTRATEGYENTRY = _FEATUREIMPUTERMETA.nested_types_by_name['ColsStrategyEntry']
_FEATUREIMPUTATIONMETA = DESCRIPTOR.message_types_by_name['FeatureImputationMeta']
FeatureImputerMeta = _reflection.GeneratedProtocolMessageType('FeatureImputerMeta', (_message.Message,), {

  'ColsStrategyEntry' : _reflection.GeneratedProtocolMessageType('ColsStrategyEntry', (_message.Message,), {
    'DESCRIPTOR' : _FEATUREIMPUTERMETA_COLSSTRATEGYENTRY,
    '__module__' : 'feature_imputation_meta_pb2'
    # @@protoc_insertion_point(class_scope:com.webank.ai.fate.core.mlmodel.buffer.FeatureImputerMeta.ColsStrategyEntry)
    })
  ,
  'DESCRIPTOR' : _FEATUREIMPUTERMETA,
  '__module__' : 'feature_imputation_meta_pb2'
  # @@protoc_insertion_point(class_scope:com.webank.ai.fate.core.mlmodel.buffer.FeatureImputerMeta)
  })
_sym_db.RegisterMessage(FeatureImputerMeta)
_sym_db.RegisterMessage(FeatureImputerMeta.ColsStrategyEntry)

FeatureImputationMeta = _reflection.GeneratedProtocolMessageType('FeatureImputationMeta', (_message.Message,), {
  'DESCRIPTOR' : _FEATUREIMPUTATIONMETA,
  '__module__' : 'feature_imputation_meta_pb2'
  # @@protoc_insertion_point(class_scope:com.webank.ai.fate.core.mlmodel.buffer.FeatureImputationMeta)
  })
_sym_db.RegisterMessage(FeatureImputationMeta)

if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'B\032FeatureImputationMetaProto'
  _FEATUREIMPUTERMETA_COLSSTRATEGYENTRY._options = None
  _FEATUREIMPUTERMETA_COLSSTRATEGYENTRY._serialized_options = b'8\001'
  _FEATUREIMPUTERMETA._serialized_start=74
  _FEATUREIMPUTERMETA._serialized_end=337
  _FEATUREIMPUTERMETA_COLSSTRATEGYENTRY._serialized_start=286
  _FEATUREIMPUTERMETA_COLSSTRATEGYENTRY._serialized_end=337
  _FEATUREIMPUTATIONMETA._serialized_start=339
  _FEATUREIMPUTATIONMETA._serialized_end=462
# @@protoc_insertion_point(module_scope)
