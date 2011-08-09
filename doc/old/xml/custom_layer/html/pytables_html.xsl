<?xml version='1.0' encoding="utf-8"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                version="1.0">

    <!-- The base stylesheet -->
    <xsl:import href="tldp-chapters.xsl"/>
    <xsl:import href="role_admon.xsl"/>
    <xsl:include href="../common/pytables_common.xsl"/>

    <!-- The CSS location -->
    <xsl:param name="html.stylesheet" select="'pytables_style.css'"></xsl:param>

    <!-- ADMONITION -->
    <!-- * use graphical admonitions in PNG format -->
    <xsl:param name="admon.graphics" select="1"></xsl:param>
    <xsl:param name="admon.graphics.extension" select="'.png'"></xsl:param>

</xsl:stylesheet>
