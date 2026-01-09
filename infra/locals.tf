locals {
  private_subnet_ids               = data.terraform_remote_state.core.outputs.private_subnet_ids
  ml_sg_id                         = data.terraform_remote_state.core.outputs.ml_security_group_id
  ecs_cluster_name                 = data.terraform_remote_state.core.outputs.ecs_cluster_name
  service_discovery_namespace_id   = data.terraform_remote_state.core.outputs.service_discovery_namespace_id
}
