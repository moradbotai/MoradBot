export const metadata = {
	title: "MoradBot Dashboard",
	description: "AI-Powered FAQ Automation for Salla Merchants",
};

export default function RootLayout({
	children,
}: {
	children: React.ReactNode;
}) {
	return (
		<html lang="ar" dir="rtl">
			<body>{children}</body>
		</html>
	);
}
