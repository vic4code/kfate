# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: sample-weight-model-param.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1fsample-weight-model-param.proto\x12&com.webank.ai.fate.core.mlmodel.buffer\"\xd8\x01\n\x16SampleWeightModelParam\x12\x0e\n\x06header\x18\x01 \x03(\t\x12\x13\n\x0bweight_mode\x18\x02 \x01(\t\x12\x65\n\x0c\x63lass_weight\x18\x03 \x03(\x0b\x32O.com.webank.ai.fate.core.mlmodel.buffer.SampleWeightModelParam.ClassWeightEntry\x1a\x32\n\x10\x43lassWeightEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x01:\x02\x38\x01\x42\x1d\x42\x1bSampleWeightModelParamProtob\x06proto3')



_SAMPLEWEIGHTMODELPARAM = DESCRIPTOR.message_types_by_name['SampleWeightModelParam']
_SAMPLEWEIGHTMODELPARAM_CLASSWEIGHTENTRY = _SAMPLEWEIGHTMODELPARAM.nested_types_by_name['ClassWeightEntry']
SampleWeightModelParam = _reflection.GeneratedProtocolMessageType('SampleWeightModelParam', (_message.Message,), {

  'ClassWeightEntry' : _reflection.GeneratedProtocolMessageType('ClassWeightEntry', (_message.Message,), {
    'DESCRIPTOR' : _SAMPLEWEIGHTMODELPARAM_CLASSWEIGHTENTRY,
    '__module__' : 'sample_weight_model_param_pb2'
    # @@protoc_insertion_point(class_scope:com.webank.ai.fate.core.mlmodel.buffer.SampleWeightModelParam.ClassWeightEntry)
    })
  ,
  'DESCRIPTOR' : _SAMPLEWEIGHTMODELPARAM,
  '__module__' : 'sample_weight_model_param_pb2'
  # @@protoc_insertion_point(class_scope:com.webank.ai.fate.core.mlmodel.buffer.SampleWeightModelParam)
  })
_sym_db.RegisterMessage(SampleWeightModelParam)
_sym_db.RegisterMessage(SampleWeightModelParam.ClassWeightEntry)

if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'B\033SampleWeightModelParamProto'
  _SAMPLEWEIGHTMODELPARAM_CLASSWEIGHTENTRY._options = None
  _SAMPLEWEIGHTMODELPARAM_CLASSWEIGHTENTRY._serialized_options = b'8\001'
  _SAMPLEWEIGHTMODELPARAM._serialized_start=76
  _SAMPLEWEIGHTMODELPARAM._serialized_end=292
  _SAMPLEWEIGHTMODELPARAM_CLASSWEIGHTENTRY._serialized_start=242
  _SAMPLEWEIGHTMODELPARAM_CLASSWEIGHTENTRY._serialized_end=292
# @@protoc_insertion_point(module_scope)
