// addInstallButton(groupId, featureId, installId, buttonText)
// groupId is SwarmUI's cleaned form of the group name ("VAE DeGrid" -> vaedegrid, "VAE Enhance" -> vaeenhance, "Film Grain" -> filmgrain).
// All three groups are served by the same node pack (this repo), so every button installs the same feature.
addInstallButton('vaedegrid', 'degrid', 'degrid', 'Install VAE DeGrid');
addInstallButton('vaeenhance', 'degrid', 'degrid', 'Install VAE DeGrid node pack');
addInstallButton('filmgrain', 'degrid', 'degrid', 'Install VAE DeGrid node pack');
