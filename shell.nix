
{ pkgs ? import <nixpkgs> {} }:
with pkgs; mkShell rec {
    nativeBuildInputs = [
        pkg-config
    ];
    buildInputs = [
        vulkan-loader # vulkan
        xorg.libXcursor xorg.libXrandr xorg.libXi # x11
        libxkbcommon wayland # wayland
    ];
    LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath buildInputs;
}

