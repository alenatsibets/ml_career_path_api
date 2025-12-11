resource "aws_ecr_repository" "ml_api" {
  name = "${var.project_name}"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Name = "${var.project_name}-ecr"
  }
}
